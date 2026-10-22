import json
import os
import shutil
import sqlite3
import uuid
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from tensorflow.keras.models import load_model

from argus_logger import infer_attack_hint
from argus_model import (
    CONFIG_PATH,
    MAX_DIAGNOSTIC_ROWS,
    MAX_TRAINING_ROWS,
    MODEL_PATH,
    SCALER_PATH,
    create_sequence,
    is_label_column,
    missing_artifacts,
    prepare_telemetry_frame,
    read_csv_temporal_sample,
    save_artifacts,
    train_from_dataframe,
)
from persistence import (
    complete_diagnostic_run,
    create_diagnostic_run,
    create_model_record,
    create_user,
    get_latest_anomaly,
    get_model,
    init_db,
    list_diagnostic_runs,
    list_latest_anomaly_per_sensor,
    list_recent_anomalies,
    list_user_models,
    log_anomaly_record,
    model_storage_dir,
    run_storage_dir,
    save_echo_answer,
    save_voice_interaction,
    list_voice_interactions,
    list_voice_incidents,
    update_voice_incident_status,
    list_bolna_executions,
    write_uploaded_file,
    authenticate_user,
)
from voice_security import issue_bolna_session_grant


st.set_page_config(page_title="Argus Echo | Industrial MLOps", layout="wide")
init_db()


@st.cache_resource
def load_system(model_id: str, model_path: str, scaler_path: str, config_path: str):
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    config = joblib.load(config_path)
    return model, scaler, config


def current_user():
    return st.session_state.get("user")


def render_auth():
    st.sidebar.subheader("Account")

    if current_user():
        st.sidebar.success(current_user()["email"])
        if st.sidebar.button("Log out"):
            st.session_state.pop("user", None)
            st.session_state.pop("selected_model_id", None)
            load_system.clear()
            st.rerun()
        return True

    mode = st.sidebar.radio("Access", ["Log in", "Create account"], horizontal=True)
    email = st.sidebar.text_input("Email")
    password = st.sidebar.text_input("Password", type="password")

    if mode == "Log in":
        if st.sidebar.button("Log in", type="primary"):
            user = authenticate_user(email, password)
            if not user:
                st.sidebar.error("Invalid email or password.")
                return False
            st.session_state["user"] = user
            st.rerun()
    else:
        if st.sidebar.button("Create account", type="primary"):
            if not email.strip() or len(password) < 8:
                st.sidebar.error("Use an email and a password with at least 8 characters.")
                return False
            try:
                st.session_state["user"] = create_user(email, password)
            except sqlite3.IntegrityError:
                st.sidebar.error("An account already exists for that email.")
                return False
            st.rerun()

    st.info("Log in or create an account to train models, run diagnostics, and view history.")
    return False


def select_model(user_id: str):
    models = list_user_models(user_id)
    if not models:
        return None, []

    selected_id = st.session_state.get("selected_model_id")
    model_ids = [model["id"] for model in models]
    if selected_id not in model_ids:
        selected_id = model_ids[0]
        st.session_state["selected_model_id"] = selected_id

    labels = {
        model["id"]: (
            f"{model['name']} | {model['raw_sensor_count']} sensors | "
            f"{model['created_at'][:19]}"
        )
        for model in models
    }
    selected_id = st.selectbox(
        "Active model",
        options=model_ids,
        index=model_ids.index(selected_id),
        format_func=lambda value: labels[value],
    )
    st.session_state["selected_model_id"] = selected_id
    return get_model(selected_id, user_id), models


def render_training_tab(user):
    st.subheader("1. Train Facility-Specific Model")
    st.write(
        "Upload normal-operation telemetry for the facility or system you want Argus "
        "to learn. The model will learn that baseline and save the runtime artifacts "
        "needed for diagnostics."
    )

    training_file = st.file_uploader(
        "Upload normal telemetry CSV",
        type=["csv"],
        key="training_csv",
    )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        model_name = st.text_input("Model name", value="Facility baseline")
    with col_b:
        epochs = st.number_input(
            "Training epochs",
            min_value=1,
            max_value=100,
            value=30,
            step=1,
        )
    with col_c:
        batch_size = st.number_input(
            "Batch size",
            min_value=8,
            max_value=256,
            value=32,
            step=8,
        )

    if training_file is None:
        models = list_user_models(user["id"])
        legacy_missing = missing_artifacts(MODEL_PATH, SCALER_PATH, CONFIG_PATH)
        if not legacy_missing:
            st.subheader("Import Existing Local Model")
            st.caption(
                "A root-level trained model exists from the earlier single-user flow. "
                "Import it into this account if you want to use it here."
            )
            if st.button("Import Existing Local Model"):
                try:
                    config = joblib.load(CONFIG_PATH)
                    model_id = uuid.uuid4().hex
                    artifact_dir = model_storage_dir(user["id"], model_id)
                    model_path = artifact_dir / "anomaly_detection_model.keras"
                    scaler_path = artifact_dir / "scaler.pkl"
                    config_path = artifact_dir / "model_config.pkl"
                    shutil.copy2(MODEL_PATH, model_path)
                    shutil.copy2(SCALER_PATH, scaler_path)
                    shutil.copy2(CONFIG_PATH, config_path)
                    saved_model = create_model_record(
                        user_id=user["id"],
                        name="Imported local model",
                        config=config,
                        training_file_path=None,
                        model_path=model_path,
                        scaler_path=scaler_path,
                        config_path=config_path,
                    )
                    st.session_state["selected_model_id"] = saved_model["id"]
                    load_system.clear()
                    st.success("Existing local model imported into this account.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Import failed. {exc}")

        if models:
            st.subheader("Your Trained Models")
            st.dataframe(
                pd.DataFrame(models)[
                    [
                        "name",
                        "raw_sensor_count",
                        "expected_feature_count",
                        "epochs",
                        "training_rows",
                        "created_at",
                    ]
                ],
                width="stretch",
            )
        return

    try:
        train_df = read_csv_temporal_sample(training_file, MAX_TRAINING_ROWS)
    except Exception as exc:
        st.error(f"Training CSV could not be sampled. {exc}")
        return
    try:
        preview_df = prepare_telemetry_frame(train_df)
    except Exception as exc:
        st.error(f"Training data could not be prepared. {exc}")
        return

    st.caption(
        f"Detected {len([c for c in preview_df.columns if '_variance' not in c.lower()])} "
        f"raw sensors and {preview_df.shape[1]} total model features."
    )
    st.caption(
        f"Using {len(train_df):,} evenly spaced rows from the full "
        f"{train_df.attrs.get('source_row_count', len(train_df)):,}-row file, preserving source order."
    )
    if any(is_label_column(column) for column in train_df.columns):
        st.info("A ground-truth label column was detected and will not be used as a sensor feature. Labelled attack rows are excluded from baseline training when the label contains normal and attack examples.")
    st.caption("Training uses validation-loss early stopping (patience 5, best weights restored); 30 epochs is the maximum starting target, not a demand to run all 30.")
    if int(epochs) < 15:
        st.warning("Fewer than 15 epochs may underfit this LSTM. Use at least 30 with early stopping for a more reliable baseline.")
    st.dataframe(preview_df.head(5), width="stretch")

    if st.button("Train Model", type="primary"):
        with st.spinner("Training Argus on uploaded normal-operation telemetry..."):
            try:
                model, scaler, config, history = train_from_dataframe(
                    train_df,
                    epochs=int(epochs),
                    batch_size=int(batch_size),
                )
                config["source_training_rows"] = int(
                    train_df.attrs.get("source_row_count", len(train_df))
                )
                config["training_sample_rows"] = len(train_df)
                config["source_sampling_method"] = "evenly_spaced_across_full_file"

                model_id = uuid.uuid4().hex
                artifact_dir = model_storage_dir(user["id"], model_id)
                training_path = None
                model_path = artifact_dir / "anomaly_detection_model.keras"
                scaler_path = artifact_dir / "scaler.pkl"
                config_path = artifact_dir / "model_config.pkl"
                save_artifacts(model, scaler, config, model_path, scaler_path, config_path)
                saved_model = create_model_record(
                    user_id=user["id"],
                    name=model_name.strip() or "Facility baseline",
                    config=config,
                    training_file_path=training_path,
                    model_path=model_path,
                    scaler_path=scaler_path,
                    config_path=config_path,
                )
                st.session_state["selected_model_id"] = saved_model["id"]
                load_system.clear()
            except Exception as exc:
                st.error(f"Training failed. {exc}")
                return

        final_loss = history.history["loss"][-1]
        st.success(
            f"Training complete. Final loss: {final_loss:.6f}. "
            f"Stopped after {config.get('trained_epochs', config['epochs'])} epochs. "
            f"Calibrated threshold: {config['anomaly_threshold']:.6f} "
            f"(held-out normal-data {config['threshold_quantile']:.1%} percentile)."
        )
        st.rerun()


def render_diagnostics_tab(user):
    st.subheader("2. Run Diagnostics")
    model_record, _ = select_model(user["id"])

    if not model_record:
        st.error("No trained model is available yet. Train a model first.")
        return

    missing = missing_artifacts(
        Path(model_record["model_path"]),
        Path(model_record["scaler_path"]),
        Path(model_record["config_path"]),
    )
    if missing:
        st.error("This model record is missing artifact files.")
        st.caption("Missing assets: " + ", ".join(str(path) for path in missing))
        return

    try:
        model, scaler, config = load_system(
            model_record["id"],
            model_record["model_path"],
            model_record["scaler_path"],
            model_record["config_path"],
        )
    except Exception as exc:
        st.error(f"Error loading model assets. {exc}")
        return

    expected_features = config["num_features"]
    time_steps = config["time_steps"]
    anomaly_threshold = config.get("anomaly_threshold")
    if anomaly_threshold is None:
        st.error(
            "This is a legacy model without a calibrated threshold. Retrain it before "
            "running diagnostics; the old fixed threshold produced unreliable results."
        )
        return
    st.sidebar.success(f"System Loaded. AI Expects {expected_features} Input Signals.")
    st.sidebar.caption(
        f"Trained on {config.get('raw_sensor_count', 'unknown')} raw sensors "
        f"for {config.get('epochs', 'unknown')} epochs."
    )
    st.sidebar.caption(
        f"Calibrated anomaly threshold: {anomaly_threshold:.6f} "
        f"(normal-data {config.get('threshold_quantile', 0.995):.1%} percentile)."
    )

    uploaded_file = st.file_uploader(
        "Upload telemetry CSV for analysis",
        type=["csv"],
        key="diagnostic_csv",
    )

    if uploaded_file is None:
        runs = list_diagnostic_runs(user["id"], n=10)
        if runs:
            st.subheader("Recent Diagnostic Runs")
            st.dataframe(pd.DataFrame(runs), width="stretch")
        return

    try:
        df = read_csv_temporal_sample(uploaded_file, MAX_DIAGNOSTIC_ROWS)
    except Exception as exc:
        st.error(f"Diagnostic CSV could not be sampled. {exc}")
        return

    try:
        process_df = prepare_telemetry_frame(df)
    except Exception as exc:
        st.error(f"Telemetry data could not be prepared. {exc}")
        return

    st.subheader("Signal Configuration")

    feature_names = list(config.get("feature_names", []))
    if feature_names:
        missing_features = [name for name in feature_names if name not in process_df.columns]
        if missing_features:
            if any(is_label_column(name) for name in missing_features):
                st.error(
                    "This saved model was trained with a ground-truth label (such as BATADAL ATT_FLAG) "
                    "as if it were a sensor. Retrain it with the updated app using Training Dataset 1, "
                    "then select the newly trained model before running diagnostics."
                )
            else:
                st.error(
                    "The diagnostic CSV is missing trained sensor features: "
                    + ", ".join(missing_features[:12])
                    + (" …" if len(missing_features) > 12 else "")
                    + ". Use the same sensor layout as the training data."
                )
            st.stop()
        # Extra numeric columns are not part of this model. Preserve its exact
        # training order so each scaler dimension continues to mean the same sensor.
        process_df = process_df.loc[:, feature_names]

    current_features = process_df.shape[1]
    if current_features != expected_features:
        st.error(
            f"SHAPE MISMATCH: The AI model expects {expected_features} signals, "
            f"but received {current_features}. Train a new model using normal data "
            "with the same sensor layout as this diagnostic CSV."
        )
        st.stop()

    st.dataframe(process_df.head(3), width="stretch")
    st.caption(
        f"Diagnostics use {len(df):,} evenly spaced rows from the full "
        f"{df.attrs.get('source_row_count', len(df)):,}-row file."
    )

    st.subheader("Edge AI Inference")

    base_sensors = [col for col in process_df.columns if "_variance" not in col.lower()]
    out_of_range = []
    scaler_feature_names = list(getattr(scaler, "feature_names_in_", feature_names))
    for sensor in base_sensors:
        if sensor not in scaler_feature_names or not hasattr(scaler, "data_min_"):
            continue
        feature_idx = scaler_feature_names.index(sensor)
        train_min = float(scaler.data_min_[feature_idx])
        train_max = float(scaler.data_max_[feature_idx])
        tolerance = 1e-6 * max(abs(train_min), abs(train_max), 1.0)
        values = process_df[sensor].to_numpy(dtype=float)
        out_mask = (values < train_min - tolerance) | (values > train_max + tolerance)
        out_fraction = float(np.mean(out_mask)) if len(out_mask) else 0.0
        if out_fraction > 0:
            out_of_range.append(
                {
                    "sensor": sensor,
                    "training_range": f"{train_min:g}–{train_max:g}",
                    "diagnostic_range": f"{float(np.min(values)):g}–{float(np.max(values)):g}",
                    "rows_outside_training_range": out_fraction,
                }
            )
    if out_of_range:
        out_of_range.sort(key=lambda item: item["rows_outside_training_range"], reverse=True)
        st.warning(
            "This telemetry is outside the model's training range for one or more sensors. "
            "A broad operating-regime shift can make nearly every row score high; treat the "
            "results as unvalidated until the model is trained on representative normal modes."
        )
        st.dataframe(
            pd.DataFrame(out_of_range).head(12),
            hide_index=True,
            width="stretch",
        )

    default_selection = base_sensors[:2] if len(base_sensors) >= 2 else base_sensors
    selected_sensors = st.multiselect(
        "Select sensors to visualize in the chart:",
        options=base_sensors,
        default=default_selection,
    )

    scaled_data = scaler.transform(process_df)
    x_input = create_sequence(scaled_data, time_steps=time_steps)

    if len(x_input) == 0:
        st.error(f"Telemetry CSV must contain at least {time_steps} rows.")
        return

    if st.button("Run Diagnostics", type="primary"):
        with st.spinner("Processing on Edge Inference Engine..."):
            run_id = uuid.uuid4().hex
            run_dir = run_storage_dir(user["id"], run_id)
            uploaded_path = None
            result_path = run_dir / "anomaly_results.json"
            create_diagnostic_run(
                user_id=user["id"],
                model_id=model_record["id"],
                uploaded_file_name=uploaded_file.name,
                uploaded_file_path=uploaded_path,
                result_path=result_path,
            )

            predictions = model.predict(x_input, batch_size=256, verbose=0)
            mae_loss = np.mean(np.abs(predictions - x_input), axis=(1, 2))
            # Attribute each anomalous window using the reconstruction error of
            # each sensor's raw and rolling-variance features.
            feature_mae = np.mean(np.abs(predictions - x_input), axis=1)

            padded_loss = np.concatenate([np.zeros(time_steps - 1), mae_loss])
            df["Anomaly_Score"] = padded_loss

            anomaly_indices = np.where(padded_loss > anomaly_threshold)[0]
            logged_count = 0
            logged_anomalies = []
            source_row_indices = df.attrs.get("source_row_indices", list(range(len(df))))
            source_row_stride = (
                float(np.median(np.diff(source_row_indices)))
                if len(source_row_indices) > 1 else 1.0
            )

            for idx in anomaly_indices:
                if idx < len(process_df):
                    row = process_df.iloc[idx]
                    feature_names = list(config.get("feature_names", []))
                    base_cols = [c for c in process_df.columns if "_variance" not in c]
                    sensor_errors = {}
                    seq_idx = idx - (time_steps - 1)
                    if 0 <= seq_idx < len(feature_mae) and feature_names:
                        for sensor in base_cols:
                            component_indices = [
                                feature_names.index(feature_name)
                                for feature_name in (sensor, f"{sensor}_variance")
                                if feature_name in feature_names
                            ]
                            if component_indices:
                                sensor_errors[sensor] = float(
                                    np.mean(feature_mae[seq_idx, component_indices])
                                )
                    if sensor_errors:
                        worst_sensor = max(sensor_errors, key=sensor_errors.get)
                    else:
                        # Older model artifacts may not include feature names.
                        var_vals = {c: row.get(f"{c}_variance", 0.0) for c in base_cols}
                        worst_sensor = max(var_vals, key=var_vals.get)
                        sensor_errors[worst_sensor] = 0.0
                    variance_val = float(row.get(f"{worst_sensor}_variance", 0.0))
                    mse_val = float(padded_loss[idx])
                    hint = infer_attack_hint(variance_val, mse_val)
                    window_start = max(0, idx - 4)
                    recent_values = [
                        round(float(value), 6)
                        for value in process_df[worst_sensor].iloc[window_start : idx + 1].tolist()
                    ]
                    source_row = df.iloc[idx]
                    time_columns = [
                        column for column in df.columns
                        if any(token in str(column).lower() for token in ("timestamp", "time", "date"))
                    ]
                    event_time = None
                    if time_columns:
                        raw_time = source_row.get(time_columns[0])
                        if pd.notna(raw_time):
                            event_time = str(raw_time)
                    location_columns = [
                        column for column in df.columns
                        if any(token in str(column).lower() for token in (
                            "site", "facility", "plant", "area", "zone", "unit",
                            "location", "asset", "equipment",
                        ))
                    ]
                    location_metadata = {
                        str(column): str(source_row.get(column))
                        for column in location_columns
                        if pd.notna(source_row.get(column)) and str(source_row.get(column)).strip()
                    }
                    telemetry_context = {
                        "current_value": recent_values[-1] if recent_values else None,
                        "recent_values": recent_values,
                        "event_time": event_time,
                        "location_metadata": location_metadata,
                        "window_rows": len(recent_values),
                        "source_row_stride": source_row_stride,
                        "source_row_indices_in_window": [
                            int(source_row_indices[window_start + offset])
                            for offset in range(len(recent_values))
                        ],
                        "variance_window": round(variance_val, 6),
                        "sensor_reconstruction_error": round(sensor_errors[worst_sensor], 6),
                        "anomaly_threshold": round(float(anomaly_threshold), 8),
                        "score_above_threshold": round(mse_val / max(float(anomaly_threshold), 1e-12), 4),
                        "sensor_error_contribution": round(
                            sensor_errors[worst_sensor] / max(sum(sensor_errors.values()), 1e-12),
                            4,
                        ),
                        "contributing_sensor_errors": {
                            name: round(value, 6)
                            for name, value in sorted(
                                sensor_errors.items(), key=lambda item: item[1], reverse=True
                            )[:5]
                        },
                        "units": "units not supplied",
                    }

                    logged_anomalies.append(
                        log_anomaly_record(
                            user_id=user["id"],
                            run_id=run_id,
                            model_id=model_record["id"],
                            sensor_id=worst_sensor,
                            facility=uploaded_file.name.replace(".csv", ""),
                            timestamp=int(source_row_indices[idx]),
                            mse_score=mse_val,
                            variance=variance_val,
                            attack_hint=hint,
                            telemetry_context=telemetry_context,
                        )
                    )
                    logged_count += 1

            result_path.write_text(
                json.dumps(
                    {
                        "model_id": model_record["id"],
                        "uploaded_file_name": uploaded_file.name,
                        "anomaly_count": logged_count,
                        "anomaly_ids": [item["id"] for item in logged_anomalies],
                    },
                    indent=2,
                )
            )
            complete_diagnostic_run(run_id, user["id"], logged_count)

            if logged_count > 0:
                st.sidebar.warning(f"{logged_count} anomalies logged -> Echo ready")
            else:
                st.sidebar.success("No anomalies detected")

            st.success("Analysis Complete.")

            scored_mask = np.arange(len(df)) >= (time_steps - 1)
            st.metric("Rows flagged", f"{logged_count:,} / {int(scored_mask.sum()):,}")
            st.caption(
                f"Flagging threshold: {anomaly_threshold:.6f}. The score is a window "
                "reconstruction error, not a probability or a diagnosis."
            )

            label_columns = [
                column for column in df.columns
                if is_label_column(column)
            ]
            if label_columns:
                labels = pd.to_numeric(df[label_columns[0]], errors="coerce")
                valid = scored_mask & labels.isin([0, 1]).to_numpy()
                if valid.any():
                    truth = labels.to_numpy(dtype=float)[valid].astype(int)
                    predicted = padded_loss[valid] > anomaly_threshold
                    tp = int(np.sum((truth == 1) & predicted))
                    fp = int(np.sum((truth == 0) & predicted))
                    fn = int(np.sum((truth == 1) & ~predicted))
                    tn = int(np.sum((truth == 0) & ~predicted))
                    normal_rows = int(np.sum(truth == 0))
                    attack_rows = int(np.sum(truth == 1))
                    if normal_rows and attack_rows:
                        precision = tp / max(tp + fp, 1)
                        recall = tp / max(tp + fn, 1)
                        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
                        false_positive_rate = fp / max(fp + tn, 1)
                        metric_cols = st.columns(4)
                        metric_cols[0].metric("Precision", f"{precision:.1%}")
                        metric_cols[1].metric("Recall", f"{recall:.1%}")
                        metric_cols[2].metric("F1", f"{f1:.1%}")
                        metric_cols[3].metric("False-positive rate", f"{false_positive_rate:.1%}")
                    elif normal_rows:
                        false_positive_rate = fp / normal_rows
                        st.error(
                            f"The scored sample contains only normal-labelled rows: "
                            f"{fp:,}/{normal_rows:,} were flagged (false-positive rate {false_positive_rate:.1%}). "
                            "Do not present these flags as confirmed anomalies; retrain using representative normal operating modes."
                        )
                    else:
                        recall = tp / max(tp + fn, 1)
                        st.metric("Attack recall on this scored sample", f"{recall:.1%}")
                    st.caption(
                        f"Evaluation against {label_columns[0]} labels on scored rows only "
                        f"(TP {tp:,}, FP {fp:,}, FN {fn:,}, TN {tn:,}; "
                        f"normal rows {normal_rows:,}, attack rows {attack_rows:,})."
                    )
                else:
                    st.info("A recognized label column was found, but its values could not be evaluated as binary 0/1 labels.")

            fig = go.Figure()

            for col in selected_sensors:
                fig.add_trace(
                    go.Scatter(x=source_row_indices, y=process_df[col], name=col, opacity=0.8)
                )
                var_col = f"{col}_variance"
                if var_col in process_df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=source_row_indices,
                            y=process_df[var_col],
                            name=var_col,
                            line=dict(dash="dot"),
                            opacity=0.5,
                        )
                    )

            fig.add_trace(
                go.Scatter(
                    x=source_row_indices,
                    y=df["Anomaly_Score"],
                    name="Window Reconstruction Error",
                    line=dict(color="red", width=2),
                    yaxis="y2",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=source_row_indices,
                    y=[anomaly_threshold] * len(df),
                    name="Calibrated threshold",
                    line=dict(color="orange", dash="dash"),
                    yaxis="y2",
                )
            )

            fig.update_layout(
                title="Multi-Sensor Telemetry & Anomaly Detection",
                yaxis=dict(title="Sensor Readings"),
                yaxis2=dict(title="Reconstruction Error", overlaying="y", side="right"),
                hovermode="x unified",
                height=600,
            )

            st.plotly_chart(fig, width="stretch")


def render_sensor_event_summary(event):
    telemetry = event.get("telemetry") or {}
    pattern_labels = {
        "replay_attack": "near_constant_signal",
        "low_entropy_anomaly": "low_variability_high_reconstruction_error",
        "high_deviation_anomaly": "high_reconstruction_error",
        "subtle_anomaly": "deviation_above_baseline",
    }
    pattern_label = pattern_labels.get(event.get("attack_hint"), event.get("attack_hint", "unknown"))
    st.markdown("**Matched sensor event**")
    cols = st.columns(4)
    cols[0].metric("Top contributing sensor", str(event.get("sensor_id", "unknown")))
    cols[1].metric("Source file / facility label", str(event.get("facility", "unknown")))
    cols[2].metric("Row", str(event.get("timestamp", "unknown")))
    cols[3].metric("Window reconstruction error", str(event.get("mse_score", "unknown")))
    st.caption(
        "Location evidence comes from the sensor tag, uploaded file/facility label, row, "
        "and any location metadata present in telemetry; Argus does not infer a site position."
    )
    if telemetry:
        st.caption(
            f"Recorded value: {telemetry.get('current_value', 'not captured')} "
            f"({telemetry.get('units', 'units unknown')}) · "
            f"Recent readings: {telemetry.get('recent_values', 'not captured')} · "
            f"Event time: {telemetry.get('event_time') or 'not supplied'} · "
            f"Location metadata: {telemetry.get('location_metadata') or 'not supplied'} · "
            f"Sensor reconstruction error: {telemetry.get('sensor_reconstruction_error', 'not captured')} · "
            f"Share of sensor error: {telemetry.get('sensor_error_contribution', 'not captured')} · "
            f"Top sensor residuals: {telemetry.get('contributing_sensor_errors', 'not captured')} · "
            f"Variance: {event.get('variance', 'unknown')} · "
            f"Pattern label: {pattern_label} (heuristic, not diagnosis)"
        )
    else:
        st.caption(
            f"Recorded variance: {event.get('variance', 'unknown')} · "
            f"Pattern label: {pattern_label} (heuristic, not diagnosis). "
            "Raw readings were not saved for this older event."
        )


def render_echo_tab(user):
    st.subheader("3. Echo Operator Guidance")

    api_key = st.text_input(
        "API key",
        type="password",
        value=os.getenv("GROQ_API_KEY", ""),
        help="Used only for Echo guidance generation in this running app session.",
    )
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key

    from echo.query_builder import build_query_from_anomaly
    from echo.rag import echo_is_ready

    if not echo_is_ready():
        st.warning("Echo knowledge base has not been built yet.")
        if st.button("Build Knowledge Base From data/", type="primary"):
            with st.spinner("Indexing ICS documents from data/..."):
                try:
                    from echo.ingest import ingest_data_folder

                    chunk_count = ingest_data_folder()
                except Exception as exc:
                    st.error(f"Knowledge base build failed. {exc}")
                    return
            st.success(f"Knowledge base ready. Indexed {chunk_count} document chunks.")
            st.rerun()
        return

    with st.expander("RAG knowledge base and source coverage"):
        st.caption(
            "Echo retrieves matching passages from data/ and supplies them to Groq. "
            "After adding or changing reference files, rebuild the index here."
        )
        if st.button("Rebuild RAG index from data/", key="echo_rebuild_index"):
            with st.spinner("Rebuilding the index from PDFs, text, DOCX, and EPANET INP files..."):
                try:
                    from echo.ingest import ingest_data_folder

                    chunk_count = ingest_data_folder()
                except Exception as exc:
                    st.error(f"Knowledge base rebuild failed. {exc}")
                    return
            st.success(f"RAG index rebuilt with {chunk_count} document chunks.")
            st.rerun()

    recent = list_recent_anomalies(user["id"], n=5)
    if not recent:
        st.info("No anomalies have been logged yet. Run diagnostics first.")
        return

    if not os.getenv("GROQ_API_KEY"):
        st.warning("API key is not set. Echo needs it to generate guidance.")

    st.dataframe(pd.DataFrame(recent), width="stretch")

    if st.button("Explain Latest Anomaly", type="primary"):
        if not os.getenv("GROQ_API_KEY"):
            st.error("Add your API key before asking Echo to explain an anomaly.")
            return

        anomaly = get_latest_anomaly(user["id"])
        if not anomaly:
            st.info("No anomaly is available for Echo to explain.")
            return

        question = build_query_from_anomaly(anomaly)
        with st.spinner("Echo is grounding the anomaly against the ICS knowledge base..."):
            try:
                from echo.rag import query_echo

                result = query_echo(question, anomaly_context=anomaly)
            except Exception as exc:
                st.error(f"Echo analysis failed. {exc}")
                return

        save_echo_answer(anomaly["id"], user["id"], result["answer"])
        st.markdown(result["answer"])
        st.caption(
            f"Response path: {result.get('model_version', 'unknown')} · "
            f"retrieval relevance {result.get('confidence', 0):.3f} "
            "(not answer correctness)"
        )
        if result.get("sources"):
            with st.expander(f"Retrieved RAG evidence · {len(result['sources'])} source passage(s)", expanded=True):
                for source in result["sources"]:
                    st.markdown(f"**{source.get('source', 'Unknown source')}**")
                    if source.get("relevance") is not None:
                        st.caption(
                            f"Lexical passage relevance {source['relevance']:.3f} "
                            "(not answer correctness)"
                        )
                    if source.get("excerpt"):
                        st.write(source["excerpt"])
        else:
            st.info("No supporting document passages were retrieved for this answer.")
        if result.get("sensor_event"):
            render_sensor_event_summary(result["sensor_event"])
        if result["escalate"] and result.get("matched_sensor"):
            st.warning(
                "This response needs human review. General ICS references do not replace "
                "an approved site procedure for this specific equipment."
            )
        else:
            st.caption(f"Sources retrieved: {result['sources_used']}")


def render_voiceops_tab(user):
    st.subheader("Voice incident assistant")
    st.caption(
        "Decision support only. Argus surfaces documented guidance and escalation paths; "
        "it does not control machinery or replace facility procedures."
    )
    from echo.ingest import ingest_data_folder
    from echo.rag import echo_is_ready, query_echo

    st.info(
        "Argus can retrieve BATADAL benchmark context, including the simulated C-Town "
        "sensor mappings. Those public references are not approved procedures for "
        "real equipment; physical actions must follow the site's approved manual or SOP."
    )

    sensor_context = list_latest_anomaly_per_sensor(user["id"], n=12)
    latest_anomaly = sensor_context[0] if sensor_context else None
    latest_anomaly_for_prompt = (
        json.dumps({"sensors": sensor_context}, ensure_ascii=False) if sensor_context else "NONE"
    )
    if latest_anomaly:
        st.info(
            f"{len(sensor_context)} recent sensor event(s) will be shared with the Bolna voice session. "
            "Ask by sensor tag to use its matching event. Most recent: "
            f"{latest_anomaly['sensor_id']} at {latest_anomaly['facility']} "
            f"(score {latest_anomaly['mse_score']})."
        )
        if any(not event.get("telemetry") for event in sensor_context):
            st.warning(
                "Some existing anomaly records predate sensor-reading capture. "
                "Re-run diagnostics on the telemetry CSV to save the readings and "
                "recent window for those sensor events."
            )
    else:
        st.warning("No anomaly is logged for this account yet. Run diagnostics first; anomaly-specific questions will escalate.")

    if not echo_is_ready():
        st.warning("The local FAISS knowledge index is not ready.")
    with st.expander("Add or update sensor manuals and approved procedures"):
        st.caption(
            "Place PDF, DOCX, TXT, or EPANET INP files in the project data/ folder, then rebuild "
            "the index here. For sensor-specific guidance, the document must identify "
            "the sensor tag or include an explicit tag-to-equipment mapping."
        )
        if st.button("Build / rebuild knowledge index from data/", key="voiceops_build_index"):
            with st.spinner("Indexing the approved procedure and reference documents..."):
                try:
                    count = ingest_data_folder()
                    st.success(f"Indexed {count} document chunks.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Index build failed: {exc}")
    if not echo_is_ready():
        return

    st.markdown("**Bolna agent integration**")
    if os.getenv("ARGUS_VOICEOPS_API_TOKEN"):
        st.success("Bolna answer endpoint is configured.")
    else:
        st.info("Bolna endpoint is ready to configure after signup. No Bolna credentials are required for this local demo.")
    public_api_url = os.getenv("ARGUS_VOICEOPS_API_URL", "").strip().rstrip("/")
    if not public_api_url:
        st.warning(
            "The browser can start a local Bolna call, but the hosted Bolna agent cannot "
            "reach localhost. Live Argus answers will work only after you provide a public "
            "HTTPS URL for port 8000 and use it in the Bolna custom function."
        )
    elif not public_api_url.lower().startswith("https://"):
        st.error("ARGUS_VOICEOPS_API_URL must use HTTPS for Bolna's hosted function call.")
    else:
        st.success("Public HTTPS Argus endpoint is configured. Confirm it routes to this app's port 8000.")

    with st.expander("Bolna setup: replace the conflicting prompt and configure the tool", expanded=True):
        st.warning(
            "Replace the existing long prompt completely. Its FAQ-only rule and scripted "
            "call branches can prevent the agent from calling Argus for equipment questions."
        )
        st.markdown("**1. Replace the Bolna system prompt with this**")
        st.code(
            """You are Argus VoiceOps, the voice interface for Argus Echo, an industrial anomaly decision-support prototype. Speak in the caller's language, calmly and in short sentences. Ask what equipment, sensor tag, alarm, or observed condition they need help with.

MANDATORY RETRIEVAL: For every equipment, sensor, alarm, anomaly, cyber incident, or action question, call get_documented_operator_guidance before answering. Do not answer these questions from memory, from this prompt, or from general knowledge. Send the caller's complete question and the full Current Argus sensor context below; do not ask the caller to repeat a sensor event already present in that context. If the context is NONE, send an empty string.

After the tool returns, answer now in this call. Turn its spoken_answer into no more than three short spoken sentences; do not read Markdown, bullets, or numbered lists aloud. Preserve the sensor/event facts, exact source name, retrieval relevance, and any human-review notice. Retrieval relevance is not answer correctness. Add no facts or steps that are absent from the tool result. If the tool reports escalate=true, no sources, or an error, say documented guidance could not be verified and direct the operator to the approved site procedure and a qualified human operator. Never say you will investigate later, call back, or get back to them.

Never invent sensor-specific procedures, diagnose a confirmed fault or attack from a heuristic score, infer a physical location, control or modify equipment, or override site procedures. Do not recommend changing PLC or network state, isolating a segment, or stopping, restarting, or adjusting equipment unless an approved procedure for that exact facility and equipment explicitly says so. BATADAL and other public benchmark documents describe a simulation; they are not real-site operating procedures. Do not collect passwords or access codes. Do not promise a ticket, callback, or response time. If asked something unrelated to industrial guidance, briefly say it is outside your scope. If the caller wants to end, close politely.

Current Argus sensor context (use only the event matching the caller's sensor): {{latest_anomaly}}""",
            language="text",
        )
        st.markdown("**2. Add this custom function in Bolna → Tools**")
        st.caption(
            "Use the exact public HTTPS host configured above. Store the Argus API token "
            "in Bolna's API token field as `Bearer <your ARGUS_VOICEOPS_API_TOKEN>`. "
            "Do not make interaction_id a required argument: Bolna does not document an "
            "execution_id prompt variable, and a web call may not have a telephony call SID."
        )
        function_url = (
            f"{public_api_url}/api/voiceops/answer"
            if public_api_url.lower().startswith("https://")
            else "https://<your-public-host>/api/voiceops/answer"
        )
        bolna_function = {
            "name": "get_documented_operator_guidance",
            "description": (
                "MANDATORY: Call for every operator question about an industrial sensor, "
                "equipment, a reading, alarm, anomaly, cyber incident, or what action to "
                "take. Call before answering. Send the complete question and the supplied "
                "Argus sensor context unchanged. Use the returned spoken_answer in this "
                "same turn; honor escalate and never promise a later response."
            ),
            "pre_call_message": "एक क्षण, मैं Argus Echo में प्रलेखित मार्गदर्शन जाँच रही हूँ।",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The operator's complete current equipment or incident question",
                    },
                    "latest_anomaly": {
                        "type": "string",
                        "description": "Copy the complete Current Argus sensor context supplied in the prompt unchanged; use an empty string if it says NONE",
                    },
                },
                "required": ["question", "latest_anomaly"],
            },
            "key": "custom_task",
            "value": {
                "method": "POST",
                "param": {
                    "question": "%(question)s",
                    "latest_anomaly": "%(latest_anomaly)s",
                },
                "url": function_url,
                "api_token": "Bearer <your ARGUS_VOICEOPS_API_TOKEN>",
                "headers": {"Content-Type": "application/json"},
            },
        }
        st.code(json.dumps(bolna_function, ensure_ascii=False, indent=2), language="json")
        st.markdown(
            "**3. Keep the current Hindi welcome and hangup prompt.** In the hangup "
            "prompt, avoid promising callbacks or saying the equipment issue is resolved. "
            "For live hosted calls, both the function URL and any Bolna webhook URL must "
            "be publicly reachable over HTTPS."
        )

    if os.getenv("BOLNA_API_KEY") and os.getenv("BOLNA_AGENT_ID"):
        st.markdown("### Talk to the Bolna voice agent")
        st.caption("The Bolna API key stays on the server. Starting a call requests a short-lived, single-use session.")
        components = st.components.v1
        api_base_url = json.dumps(os.getenv("ARGUS_VOICEOPS_API_URL", "").rstrip("/"))
        session_grant = json.dumps(
            issue_bolna_session_grant(user["id"])
            if os.getenv("ARGUS_VOICEOPS_API_TOKEN") else ""
        )
        voice_widget = """
            <div style="font-family: sans-serif; padding: 8px; border: 1px solid #334155; border-radius: 12px;">
              <button id="start" style="padding: 10px 16px;">Start voice session</button>
              <button id="stop" style="padding: 10px 16px;" disabled>End session</button>
              <span id="state" style="margin-left: 12px;">Ready</span>
              <div id="error" style="color:#b91c1c; margin-top:8px;"></div>
            </div>
            <script src="https://cdn.jsdelivr.net/gh/bolna-ai/web-call@v3.0.0/dist/bolna-web-call.min.js"></script>
            <script>
              const start = document.getElementById('start');
              const stop = document.getElementById('stop');
              const state = document.getElementById('state');
              const error = document.getElementById('error');
              let call = null;
              start.onclick = async () => {
                error.textContent = '';
                start.disabled = true;
                try {
                  const configuredApiBase = __ARGUS_API_BASE_URL__;
                  let appOrigin = '';
                  try { appOrigin = window.parent.location.origin; } catch (e) {}
                  if (!appOrigin && document.referrer) {
                    try { appOrigin = new URL(document.referrer).origin; } catch (e) {}
                  }
                  let apiBase = configuredApiBase;
                  if (!apiBase && appOrigin) {
                    const apiUrl = new URL(appOrigin);
                    apiUrl.port = '8000';
                    apiBase = apiUrl.origin;
                  }
                  if (!apiBase) {
                    throw new Error('Could not determine the Argus API address. Set ARGUS_VOICEOPS_API_URL.');
                  }
                  call = new window.BolnaWebCall({sessionUrl: `${apiBase}/api/bolna-session`});
                  call.on('state-change', value => state.textContent = value);
                  call.on('call-end', () => { start.disabled = false; stop.disabled = true; });
                  call.on('error', e => { error.textContent = e.message || e.code || 'Call failed'; });
                  await call.start({userData: {latest_anomaly: __LATEST_ANOMALY__, _argus_session_grant: __SESSION_GRANT__}});
                  stop.disabled = false;
                } catch (e) {
                  error.textContent = e.message || 'Could not start the call';
                  start.disabled = false;
                }
              };
              stop.onclick = async () => {
                if (call) await call.stop();
                start.disabled = false;
                stop.disabled = true;
              };
            </script>
            """.replace("__ARGUS_API_BASE_URL__", api_base_url).replace(
                "__LATEST_ANOMALY__", json.dumps(latest_anomaly_for_prompt)
            ).replace("__SESSION_GRANT__", session_grant)
        components.html(
            voice_widget,
            height=110,
            scrolling=False,
        )
    else:
        st.info("After Bolna signup, set BOLNA_API_KEY and BOLNA_AGENT_ID to enable live browser voice here.")

    with st.form("voiceops_question_form"):
        question = st.text_area(
            "Operator question (or paste a Bolna transcript)",
            placeholder="The pump vibration just spiked—what should I check?",
            max_chars=2000,
        )
        interaction_id = st.text_input("Bolna interaction ID (optional)", max_chars=256)
        submitted = st.form_submit_button("Ask VoiceOps", type="primary")
    if submitted:
        if not question.strip():
            st.error("Enter or transcribe a question first.")
        else:
            current_sensor_context = {"sensors": sensor_context} if sensor_context else None
            if not os.getenv("GROQ_API_KEY"):
                result = {
                    "answer": (
                        "I could not generate a grounded response because the language-model "
                        "service is not configured. Please follow the approved site procedure "
                        "and contact a qualified human operator."
                    ),
                    "sources": [],
                    "sources_used": 0,
                    "confidence": 0.0,
                    "confidence_label": "guidance unavailable",
                    "escalate": True,
                    "model_version": "voiceops-fallback",
                    "sensor_event": None,
                }
            else:
                with st.spinner("Retrieving documented guidance and checking evidence..."):
                    try:
                        result = query_echo(
                            question,
                            anomaly_context=current_sensor_context,
                            voice_mode=True,
                        )
                    except Exception:
                        result = {
                            "answer": (
                                "I could not retrieve documented guidance for this request. "
                                "Please follow the approved site procedure and contact a "
                                "qualified human operator."
                            ),
                            "sources": [],
                            "sources_used": 0,
                            "confidence": 0.0,
                            "confidence_label": "guidance unavailable",
                            "escalate": True,
                            "model_version": "voiceops-fallback",
                            "sensor_event": None,
                        }
            save_voice_interaction(
                question.strip(), question.strip(), result,
                interaction_id.strip() or None,
                anomaly_context=current_sensor_context,
            )
            if result["escalate"]:
                st.error("A human review incident was created. The available response may not include a site-approved procedure for this equipment.")
            else:
                st.success("Documented guidance found. Verify against facility procedures.")
            st.markdown(f"**Response:** {result['answer']}")
            if result.get("sensor_event"):
                render_sensor_event_summary(result["sensor_event"])
            st.metric("Retrieval confidence", f"{result['confidence']:.0%}")
            st.caption("This measures retrieval relevance, not answer correctness.")
            st.caption(f"Model version: {result['model_version']}")
            for source in result["sources"]:
                st.markdown(f"**{source['source']}** · chunk {source['chunk']}")
                st.caption(source["excerpt"])
            if not result["escalate"]:
                st.markdown("**Speak this response through the configured Bolna agent.**")

    st.divider()
    incidents = list_voice_incidents()
    st.markdown("### Human review incidents")
    if not incidents:
        st.info("No human-review incidents have been created.")
    else:
        for incident in incidents:
            with st.expander(f"{incident['created_at'][:19]} · {incident['status']} · {incident['question'][:90]}"):
                st.write(incident["answer"])
                st.caption(f"Retrieval confidence: {incident['confidence']:.0%} · Model: {incident['model_version']}")
                status = st.selectbox(
                    "Review status", ["open", "acknowledged", "resolved"],
                    index=["open", "acknowledged", "resolved"].index(incident["status"]),
                    key=f"incident_status_{incident['id']}",
                )
                if st.button("Update status", key=f"incident_update_{incident['id']}"):
                    update_voice_incident_status(incident["id"], status)
                    st.rerun()
                for source in incident["sources"]:
                    st.caption(f"{source['source']}: {source['excerpt']}")

    with st.expander("Recent VoiceOps quality log"):
        logs = list_voice_interactions()
        if logs:
            st.dataframe([
                {"time": row["created_at"], "question": row["question"],
                 "confidence": row["confidence"], "escalated": bool(row["escalated"]),
                 "model": row["model_version"], "interaction_id": row["interaction_id"]}
                for row in logs
            ], use_container_width=True)
        else:
            st.info("No VoiceOps questions logged yet.")

    with st.expander("Bolna call execution history"):
        executions = list_bolna_executions()
        if executions:
            st.dataframe([
                {"updated": row["updated_at"], "execution": row["execution_id"],
                 "status": row["status"], "duration_seconds": row["conversation_duration"],
                 "cost": row["total_cost"], "summary": row["summary"],
                 "transcript": row["transcript"]}
                for row in executions
            ], use_container_width=True)
        else:
            st.info("No Bolna executions received yet.")


st.title("Argus Echo")
st.markdown(
    """
Industrial telemetry anomaly detection with a complete model-to-deployment workflow.

**AI/ML:** Train an LSTM autoencoder on normal telemetry and investigate reconstruction-error anomalies.

**MLOps / DevOps:** Persist model artifacts, run diagnostics, and deploy the Streamlit service with Docker and Kubernetes.
"""
)

if not render_auth():
    st.stop()

user = current_user()
train_tab, diagnostics_tab, echo_tab, voiceops_tab = st.tabs(
    ["Train Model", "Run Diagnostics", "Echo Guidance", "VoiceOps"]
)

with train_tab:
    render_training_tab(user)

with diagnostics_tab:
    render_diagnostics_tab(user)

with echo_tab:
    render_echo_tab(user)

with voiceops_tab:
    render_voiceops_tab(user)
