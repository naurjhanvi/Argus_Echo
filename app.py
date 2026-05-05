import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import plotly.graph_objects as go
from argus_logger import log_anomaly, infer_attack_hint  

st.set_page_config(page_title="Universal Edge AI Diagnostic", layout="wide")
st.title("Universal Edge AI Diagnostic Tool")
st.markdown("""
**System Status:** Ready for Analysis
**Target Domain:** Safety-Critical Infrastructure (ICS, Aerospace, Power Grids)
""")

@st.cache_resource
def load_system():
    model = load_model("anomaly_detection_model.keras")
    scaler = joblib.load("scaler.pkl")
    config = joblib.load("model_config.pkl")
    return model, scaler, config

try:
    model, scaler, config = load_system()
    EXPECTED_FEATURES = config['num_features']
    TIME_STEPS = config['time_steps']
    st.sidebar.success(f"System Loaded. AI Expects {EXPECTED_FEATURES} Input Signals.")
except Exception as e:
    st.error(f"Error loading model assets. {e}")
    st.stop()

def create_sequence(data, time_steps):
    sequence = []
    for i in range(len(data) - time_steps):
        sequence.append(data[i:(i + time_steps)])
    return np.array(sequence)

uploaded_file = st.file_uploader("Upload Telemetry Log (CSV)", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    drop_keywords = ['time', 'timestamp', 'date', 'label']
    cols_to_drop = [c for c in df.columns if any(keyword in c.lower() for keyword in drop_keywords)]
    process_df = df.drop(columns=cols_to_drop)
    process_df = process_df.select_dtypes(include=[np.number])
    
    st.subheader("1. Signal Configuration")
    
    ROLLING_WINDOW = 5
    ANOMALY_THRESHOLD = 0.13
    original_cols = list(process_df.columns)
    
    for col in original_cols:
        variance_col_name = f"{col}_variance"
        process_df[variance_col_name] = process_df[col].rolling(window=ROLLING_WINDOW).var()
    
    process_df = process_df.fillna(0)
    
    current_features = process_df.shape[1]
    if current_features != EXPECTED_FEATURES:
        st.error(f"SHAPE MISMATCH: The AI model expects {EXPECTED_FEATURES} signals, but received {current_features}.")
        st.stop()
        
    st.dataframe(process_df.head(3), use_container_width=True)

    st.subheader("2. Edge AI Inference")
    
    base_sensors = [col for col in process_df.columns if "_variance" not in col.lower()]
    default_selection = base_sensors[:2] if len(base_sensors) >= 2 else base_sensors
    selected_sensors = st.multiselect(
        "Select sensors to visualize in the chart:",
        options=base_sensors,
        default=default_selection
    )
    
    scaled_data = scaler.transform(process_df)
    X_input = create_sequence(scaled_data, time_steps=TIME_STEPS)
    
    if st.button("Run Diagnostics"):
        with st.spinner("Processing on Edge Inference Engine..."):
            predictions = model.predict(X_input)
            mae_loss = np.mean(np.abs(predictions - X_input), axis=(1, 2))
            
            padded_loss = np.concatenate([np.zeros(TIME_STEPS), mae_loss])
            df['Anomaly_Score'] = padded_loss

            anomaly_indices = np.where(padded_loss > ANOMALY_THRESHOLD)[0]
            logged_count = 0

            for idx in anomaly_indices:
                if idx < len(process_df):
                    row = process_df.iloc[idx]
                    base_cols = [c for c in process_df.columns if "_variance" not in c]
                    worst_sensor = base_cols[0] 
                    
                    var_vals = {
                        c: row.get(f"{c}_variance", 0.0)
                        for c in base_cols
                    }
                    worst_sensor = max(var_vals, key=var_vals.get)
                    variance_val = float(var_vals[worst_sensor])
                    mse_val = float(padded_loss[idx])
                    hint = infer_attack_hint(variance_val, mse_val)

                    log_anomaly(
                        sensor_id=worst_sensor,
                        facility=uploaded_file.name.replace(".csv", ""),
                        timestamp=int(idx),
                        mse_score=mse_val,
                        variance=variance_val,
                        attack_hint=hint
                    )
                    logged_count += 1

            if logged_count > 0:
                st.sidebar.warning(f"⚠ {logged_count} anomalies logged → Echo ready")
            else:
                st.sidebar.success("✓ No anomalies detected")

            st.success("Analysis Complete.")
            
            fig = go.Figure()
            
            for col in selected_sensors:
                fig.add_trace(go.Scatter(
                    x=df.index, y=process_df[col],
                    name=col, opacity=0.8
                ))
                var_col = f"{col}_variance"
                if var_col in process_df.columns:
                    fig.add_trace(go.Scatter(
                        x=df.index, y=process_df[var_col],
                        name=var_col,
                        line=dict(dash='dot'),
                        opacity=0.5
                    ))
            
            fig.add_trace(go.Scatter(
                x=df.index, y=df['Anomaly_Score'],
                name='Anomaly Score',
                line=dict(color='red', width=2),
                yaxis='y2'
            ))
            
            fig.update_layout(
                title="Multi-Sensor Telemetry & Anomaly Detection",
                yaxis=dict(title="Sensor Readings"),
                yaxis2=dict(title="Anomaly Probability", overlaying='y', side='right'),
                hovermode="x unified",
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)