import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Input, RepeatVector, TimeDistributed
from tensorflow.keras.models import Sequential


MODEL_PATH = Path("anomaly_detection_model.keras")
SCALER_PATH = Path("scaler.pkl")
CONFIG_PATH = Path("model_config.pkl")
DROP_KEYWORDS = ("time", "timestamp", "date", "label", "attack", "target", "class", "att_flag")
LABEL_COLUMN_NAMES = {
    "attack",
    "is_attack",
    "attack_label",
    "anomaly",
    "anomaly_label",
    "att_flag",  # BATADAL ground-truth label
}
ROLLING_WINDOW = 5
TIME_STEPS = 10
THRESHOLD_QUANTILE = 0.995
MAX_TRAINING_ROWS = int(os.getenv("ARGUS_MAX_TRAINING_ROWS", "20000"))
MAX_DIAGNOSTIC_ROWS = int(os.getenv("ARGUS_MAX_DIAGNOSTIC_ROWS", "20000"))


def is_label_column(column) -> bool:
    """Return whether a CSV column is a ground-truth label, not a sensor."""
    return str(column).strip().lower() in LABEL_COLUMN_NAMES


def read_csv_temporal_sample(source, max_rows: int, chunksize: int = 50_000) -> pd.DataFrame:
    """Read an evenly spaced, order-preserving sample across the entire CSV.

    A first-rows-only cap can miss later operating modes and attack intervals.
    Two bounded-memory passes let training and diagnostics cover the full file.
    The original source row for each retained row is saved in DataFrame.attrs.
    """
    if max_rows < 1:
        raise ValueError("max_rows must be positive.")
    if not hasattr(source, "seek"):
        raise ValueError("Uploaded CSV must support seeking for full-file sampling.")

    source.seek(0)
    total_rows = 0
    try:
        for chunk in pd.read_csv(source, chunksize=chunksize):
            total_rows += len(chunk)
        if total_rows == 0:
            raise ValueError("The uploaded CSV contains no data rows.")

        sample_size = min(total_rows, max_rows)
        source_row_indices = np.linspace(0, total_rows - 1, sample_size, dtype=np.int64)
        if total_rows <= max_rows:
            source.seek(0)
            sampled = pd.read_csv(source)
        else:
            source.seek(0)
            selected_chunks = []
            offset = 0
            position = 0
            for chunk in pd.read_csv(source, chunksize=chunksize):
                end = offset + len(chunk)
                start_position = position
                while position < sample_size and source_row_indices[position] < end:
                    position += 1
                if position > start_position:
                    local_positions = source_row_indices[start_position:position] - offset
                    selected_chunks.append(chunk.iloc[local_positions])
                offset = end
            sampled = pd.concat(selected_chunks, ignore_index=True)
        sampled = sampled.reset_index(drop=True)
        sampled.attrs["source_row_indices"] = source_row_indices.tolist()
        sampled.attrs["source_row_count"] = total_rows
        return sampled
    finally:
        source.seek(0)


def prepare_telemetry_frame(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [
        col
        for col in df.columns
        if is_label_column(col)
        or any(keyword in str(col).strip().lower() for keyword in DROP_KEYWORDS)
    ]
    data = df.drop(columns=cols_to_drop)
    data = data.select_dtypes(include=[np.number])

    if data.empty:
        raise ValueError("No numeric telemetry columns were found in the uploaded CSV.")

    original_cols = list(data.columns)
    for col in original_cols:
        data[f"{col}_variance"] = data[col].rolling(window=ROLLING_WINDOW).var()

    return data.fillna(0).astype("float32")


def create_sequence(dataset, time_steps: int = TIME_STEPS):
    dataset = np.asarray(dataset, dtype=np.float32)
    if len(dataset) < time_steps:
        return np.empty((0, time_steps, dataset.shape[1]), dtype=np.float32)

    # Include the final complete window so its reconstruction score aligns with
    # the final telemetry row instead of silently omitting that row.
    shape = (len(dataset) - time_steps + 1, time_steps, dataset.shape[1])
    strides = (dataset.strides[0], dataset.strides[0], dataset.strides[1])
    return np.lib.stride_tricks.as_strided(dataset, shape=shape, strides=strides).copy()


def _normal_training_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove explicitly labelled attack rows before learning the normal baseline."""
    attack_columns = [
        col for col in df.columns
        if is_label_column(col)
    ]
    if not attack_columns:
        return df, 0

    label = df[attack_columns[0]]
    numeric = pd.to_numeric(label, errors="coerce")
    if numeric.notna().all() and set(numeric.unique()).issubset({0, 1}):
        keep = numeric == 0
    else:
        normalized = label.astype(str).str.strip().str.lower()
        known = {"normal", "benign", "0", "false", "no"}
        anomalous = {"attack", "anomaly", "anomalous", "malicious", "1", "true", "yes"}
        values = set(normalized.unique())
        if values <= known | anomalous and values & known and values & anomalous:
            keep = normalized.isin(known)
        else:
            return df, 0

    filtered = df.loc[keep].copy()
    excluded = len(df) - len(filtered)
    if filtered.empty:
        raise ValueError("Training data has no rows labelled as normal (Attack=0).")
    return filtered, excluded


def build_model(time_steps: int, num_features: int):
    model = Sequential(
        [
            Input(shape=(time_steps, num_features)),
            LSTM(
                16,
                activation="relu",
                return_sequences=False,
            ),
            RepeatVector(time_steps),
            LSTM(16, activation="relu", return_sequences=True),
            TimeDistributed(Dense(num_features)),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def train_from_dataframe(
    df: pd.DataFrame,
    epochs: int = 50,
    batch_size: int = 32,
    validation_split: float = 0.1,
):
    original_rows = len(df)
    df, labelled_attack_rows = _normal_training_rows(df)
    if len(df) > MAX_TRAINING_ROWS:
        step = max(len(df) // MAX_TRAINING_ROWS, 1)
        df = df.iloc[::step].head(MAX_TRAINING_ROWS).copy()

    prepared = prepare_telemetry_frame(df)

    if len(prepared) <= TIME_STEPS:
        raise ValueError(
            f"Training data must contain more than {TIME_STEPS} rows after preprocessing."
        )

    # Keep the final tenth of the normal data out of model fitting. Use it only
    # to calibrate a threshold for this exact model and facility baseline.
    split_at = max(int(len(prepared) * 0.9), TIME_STEPS + 1)
    if len(prepared) - split_at < TIME_STEPS:
        split_at = len(prepared) - TIME_STEPS
    if split_at <= TIME_STEPS or split_at >= len(prepared):
        raise ValueError("Training data needs enough normal rows for training and threshold calibration.")

    scaler = MinMaxScaler()
    scaler.fit(prepared.iloc[:split_at])
    scaled_data = scaler.transform(prepared).astype("float32")
    x_train = create_sequence(scaled_data[:split_at], TIME_STEPS)
    num_features = x_train.shape[2]
    validation_start = split_at - (TIME_STEPS - 1)
    x_validation = create_sequence(scaled_data[validation_start:], TIME_STEPS)

    tf.keras.backend.clear_session()
    model = build_model(TIME_STEPS, num_features)
    history = model.fit(
        x_train,
        x_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_validation, x_validation),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                min_delta=1e-4,
                restore_best_weights=True,
            )
        ],
        verbose=0,
    )

    validation_predictions = model.predict(x_validation, batch_size=256, verbose=0)
    validation_scores = np.mean(np.abs(validation_predictions - x_validation), axis=(1, 2))
    anomaly_threshold = float(np.quantile(validation_scores, THRESHOLD_QUANTILE))

    config = {
        "time_steps": TIME_STEPS,
        "num_features": num_features,
        "feature_names": list(prepared.columns),
        "raw_sensor_count": len([c for c in prepared.columns if "_variance" not in c.lower()]),
        "rolling_window": ROLLING_WINDOW,
        "training_rows": len(prepared),
        "source_training_rows": original_rows,
        "max_training_rows": MAX_TRAINING_ROWS,
        "epochs": epochs,
        "trained_epochs": len(history.history.get("loss", [])),
        "early_stopping_patience": 5,
        "batch_size": batch_size,
        "anomaly_threshold": anomaly_threshold,
        "threshold_quantile": THRESHOLD_QUANTILE,
        "threshold_calibration_rows": len(x_validation),
        "threshold_calibration_false_positive_target": 1.0 - THRESHOLD_QUANTILE,
        "labelled_attack_rows_excluded": labelled_attack_rows,
    }

    return model, scaler, config, history


def save_artifacts(
    model,
    scaler,
    config,
    model_path: Path = MODEL_PATH,
    scaler_path: Path = SCALER_PATH,
    config_path: Path = CONFIG_PATH,
):
    model_path.parent.mkdir(parents=True, exist_ok=True)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(config, config_path)


def missing_artifacts(
    model_path: Path = MODEL_PATH,
    scaler_path: Path = SCALER_PATH,
    config_path: Path = CONFIG_PATH,
):
    return [path for path in (model_path, scaler_path, config_path) if not path.exists()]
