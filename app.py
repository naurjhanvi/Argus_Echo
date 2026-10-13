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
    missing_artifacts,
    prepare_telemetry_frame,
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
    list_recent_anomalies,
    list_user_models,
    log_anomaly_record,
    model_storage_dir,
    run_storage_dir,
    save_echo_answer,
    write_uploaded_file,
    authenticate_user,
)


ANOMALY_THRESHOLD = 0.13

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
            value=15,
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

    train_df = pd.read_csv(training_file, nrows=MAX_TRAINING_ROWS)
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
        f"Training preview and model fitting use up to {MAX_TRAINING_ROWS:,} rows "
        "to keep the live service responsive."
    )
    st.dataframe(preview_df.head(5), width="stretch")

    if st.button("Train Model", type="primary"):
        with st.spinner("Training Argus on uploaded normal-operation telemetry..."):
            try:
                model, scaler, config, history = train_from_dataframe(
                    train_df,
                    epochs=int(epochs),
                    batch_size=int(batch_size),
                )

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
        st.success(f"Training complete. Final loss: {final_loss:.6f}")
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
    st.sidebar.success(f"System Loaded. AI Expects {expected_features} Input Signals.")
    st.sidebar.caption(
        f"Trained on {config.get('raw_sensor_count', 'unknown')} raw sensors "
        f"for {config.get('epochs', 'unknown')} epochs."
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

    df = pd.read_csv(uploaded_file, nrows=MAX_DIAGNOSTIC_ROWS)

    try:
        process_df = prepare_telemetry_frame(df)
    except Exception as exc:
        st.error(f"Telemetry data could not be prepared. {exc}")
        return

    st.subheader("Signal Configuration")

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
        f"Diagnostics use up to {MAX_DIAGNOSTIC_ROWS:,} rows per uploaded CSV "
        "to keep the live service responsive."
    )

    st.subheader("Edge AI Inference")

    base_sensors = [col for col in process_df.columns if "_variance" not in col.lower()]
    default_selection = base_sensors[:2] if len(base_sensors) >= 2 else base_sensors
    selected_sensors = st.multiselect(
        "Select sensors to visualize in the chart:",
        options=base_sensors,
        default=default_selection,
    )

    scaled_data = scaler.transform(process_df)
    x_input = create_sequence(scaled_data, time_steps=time_steps)

    if len(x_input) == 0:
        st.error(f"Telemetry CSV must contain more than {time_steps} rows.")
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

            padded_loss = np.concatenate([np.zeros(time_steps), mae_loss])
            df["Anomaly_Score"] = padded_loss

            anomaly_indices = np.where(padded_loss > ANOMALY_THRESHOLD)[0]
            logged_count = 0
            logged_anomalies = []

            for idx in anomaly_indices:
                if idx < len(process_df):
                    row = process_df.iloc[idx]
                    base_cols = [c for c in process_df.columns if "_variance" not in c]
                    var_vals = {c: row.get(f"{c}_variance", 0.0) for c in base_cols}
                    worst_sensor = max(var_vals, key=var_vals.get)
                    variance_val = float(var_vals[worst_sensor])
                    mse_val = float(padded_loss[idx])
                    hint = infer_attack_hint(variance_val, mse_val)

                    logged_anomalies.append(
                        log_anomaly_record(
                            user_id=user["id"],
                            run_id=run_id,
                            model_id=model_record["id"],
                            sensor_id=worst_sensor,
                            facility=uploaded_file.name.replace(".csv", ""),
                            timestamp=int(idx),
                            mse_score=mse_val,
                            variance=variance_val,
                            attack_hint=hint,
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

            fig = go.Figure()

            for col in selected_sensors:
                fig.add_trace(
                    go.Scatter(x=df.index, y=process_df[col], name=col, opacity=0.8)
                )
                var_col = f"{col}_variance"
                if var_col in process_df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=process_df[var_col],
                            name=var_col,
                            line=dict(dash="dot"),
                            opacity=0.5,
                        )
                    )

            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["Anomaly_Score"],
                    name="Anomaly Score",
                    line=dict(color="red", width=2),
                    yaxis="y2",
                )
            )

            fig.update_layout(
                title="Multi-Sensor Telemetry & Anomaly Detection",
                yaxis=dict(title="Sensor Readings"),
                yaxis2=dict(title="Anomaly Probability", overlaying="y", side="right"),
                hovermode="x unified",
                height=600,
            )

            st.plotly_chart(fig, width="stretch")


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

                result = query_echo(question)
            except Exception as exc:
                st.error(f"Echo analysis failed. {exc}")
                return

        save_echo_answer(anomaly["id"], user["id"], result["answer"])
        st.markdown(result["answer"])
        st.caption(f"Sources retrieved: {result['sources_used']}")


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
train_tab, diagnostics_tab, echo_tab = st.tabs(
    ["Train Model", "Run Diagnostics", "Echo Guidance"]
)

with train_tab:
    render_training_tab(user)

with diagnostics_tab:
    render_diagnostics_tab(user)

with echo_tab:
    render_echo_tab(user)
