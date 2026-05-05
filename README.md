# 👁 Argus Echo
### Edge AI Anomaly Detection + RAG-Powered Operator Guidance for ICS Security

**Argus detects. Echo explains.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.20+-red)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-green)](https://langchain.com)
[![Groq](https://img.shields.io/badge/LLM-Groq%20Llama%203.3-purple)](https://groq.com)

---

## Overview

Argus Echo was originally scoped as a nuclear power plant intrusion detection system. The core problem: sophisticated state-sponsored actors execute **Replay Attacks** on ICS infrastructure — freezing a sensor's output to broadcast a perfectly normal baseline to the control room while physically sabotaging the machinery in reality. Standard time-series AI fails to catch this because a flatline has near-zero reconstruction error.

**The constraint that shaped the architecture:** Nuclear plant SCADA telemetry is classified. No public dataset exists. Rather than abandoning the research, the system was redesigned as a **domain-agnostic Edge AI framework** — mathematically rigorous enough for nuclear-grade ICS security, but deployable against any operator's telemetry.

**For this public deployment**, Argus Echo is trained and validated against the **HAI 22.04 dataset** — a hardware-in-the-loop simulated steam-turbine and pumped-storage hydropower system developed by researchers at the Affiliated Institute of ETRI. It is the most rigorous publicly available ICS security benchmark and serves as a direct proxy for the kind of telemetry found in safety-critical infrastructure.

> A nuclear plant operator would deploy their own instance, train on their SCADA normal-operation data, and get a model specific to their facility. The HAI deployment demonstrates the pipeline on real ICS data — the architecture is identical, only the training data changes.

---

## The Core Technical Problem

Standard LSTM Autoencoders detect anomalies by measuring reconstruction error (MSE) between input and predicted sequences. This works for spikes and deviations — but fails catastrophically for Replay Attacks:

```
Replay Attack → sensor output frozen → perfect flatline
LSTM-AE sees flatline → reconstructs it perfectly → MSE ≈ 0
System reports: NO ANOMALY  ← catastrophic failure
```

This is the **Low-Entropy Blind Spot** — the model inherently trusts perfectly static data.

**Argus solves this mathematically, not by scaling the model:**

```python
# Before tensor sequence creation, compute rolling variance per sensor
σ² = (1/N-1) Σ(xᵢ - x̄)²

# When a sensor is spoofed and flatlines:
# variance → exactly 0.0
# In a thermodynamic system, 0.0 variance is a physical impossibility
# The ~3,300 parameter LSTM-AE flags this as a massive anomaly
```

The computational burden shifts from the deep learning layer to a deterministic feature engineering layer — keeping the model lightweight enough for Edge microprocessor deployment.

---

## Architecture

```
HAI 22.04 Telemetry (87 sensors → 174 features with variance)
        ↓
Argus LSTM-AE Inference
        ↓
Anomaly Detected (MAE > 0.13 threshold)
        ↓
Structured JSON Anomaly Log {sensor_id, mse_score, variance}
        ↓
LangChain RAG Pipeline + FAISS Vector Store (ICS Security Corpus)
        ↓
Groq Llama 3.3 70B → Grounded Operator Guidance
        ↓
RAGAS Scoring (Faithfulness / Relevancy)
        ↓
Argus Echo Unified Dashboard (Streamlit)
```

### RAGAS Faithfulness as an Operational Trust Indicator

Every Echo interpretation is automatically scored. The faithfulness score is repurposed from an academic LLM evaluation metric into an **operational safety feature**:

| Faithfulness | Confidence | Operator Action |
|---|---|---|
| ≥ 0.85 | HIGH | Interpretation grounded in ICS literature — act on it |
| 0.60–0.85 | MODERATE | Partially inferred — verify before acting |
| < 0.60 | LOW | Insufficient documentation — escalate to security team immediately |

---

## Stack

| Layer | Technology |
|---|---|
| Detection Engine | LSTM Autoencoder (TensorFlow/Keras, ~3,300 params) |
| Feature Engineering | Multivariate Rolling Variance (NumPy/Pandas) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store | FAISS |
| RAG Orchestration | LangChain |
| LLM | Groq — Llama 3.3 70B Versatile |
| LLM Evaluation | RAGAS (faithfulness, answer relevancy, context precision) |
| Dashboard | Streamlit + Plotly |

---

## Project Structure

```
argus_echo/
├── app.py                    ← Unified Streamlit Application
├── train_model.py            ← LSTM-AE training script
├── anomaly_detection_model.keras ← Trained model
├── requirements.txt          ← Python dependencies
├── .env                      ← Environment variables
├── .gitignore                ← Git ignore rules
│
├── echo/
│   ├── __init__.py
│   ├── ingest.py             ← ICS corpus ingestion → FAISS
│   ├── rag.py                ← ICS analyst prompt + RAG chain
│   ├── evaluate.py           ← RAGAS scoring
│   └── query_builder.py      ← Anomaly JSON → LLM query
│
├── data/                     ← ICS security corpus (PDFs)
└── vectorstore/              ← FAISS index (auto-generated)
```

---

## Local Setup

### Prerequisites

- Python 3.10+
- Groq API key — free at [console.groq.com](https://console.groq.com)
- HAI 22.04 dataset — [icsdataset/hai](https://github.com/icsdataset/hai)

### Installation

```bash
git clone https://github.com/naurjhanvi/argus-echo
cd argus_echo

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
```

### Environment

```bash
# .env
GROQ_API_KEY=your_groq_api_key_here
```

### Train on HAI 22.04

Download the HAI 22.04 dataset. Use any normal-operation CSV for training:

```bash
python train_model.py path/to/hai_normal.csv
```

This generates `anomaly_detection_model.keras`, `scaler.pkl`, and `model_config.pkl`. The model automatically reads the shape of your CSV — 87 HAI sensors become 174 features (87 raw + 87 variance). No hardcoded sensor names.

### Build Echo Knowledge Base

Add ICS security PDFs to `data/` then:

```bash
python -c "from echo.ingest import ingest_data_folder; print(ingest_data_folder())"
```

### Run Locally

```bash
streamlit run app.py
```

---

## ICS Security Corpus

Echo's interpretation quality is directly proportional to corpus quality.

| Document | Status |
|---|---|
| NIST SP 800-82 Rev 3 — ICS Security Guide | ✅ Included |
| Malhotra et al. (2015) — LSTM for Anomaly Detection | ✅ Included |
| Project Argus Technical Report | ✅ Included |
| Kravchik & Shabtai (2018) — Detecting Cyber Attacks in ICS | ⬇ Recommended |
| CISA ICS Recommended Practices | ⬇ Recommended |

> Adding Kravchik & Shabtai (2018) alone will significantly improve faithfulness scores — it covers exactly the attack patterns Argus detects and is directly cited in the Argus technical report.

---

## Deployment Philosophy & Future Work

### Why a Local MVP?
Argus Echo is intentionally scoped as a local Streamlit application for its MVP phase. Because this system is designed for classified, safety-critical ICS environments (like nuclear power plants), relying on external cloud APIs and public web deployments introduces unacceptable vulnerabilities. The primary focus of this build is proving the rigor of the **LSTM anomaly detection logic** and the **RAG-based operator guidance pipelines**. 

### The Path to "True Edge"
The target architecture for production will bypass cloud deployment entirely in favor of a true, air-gapped Edge deployment:
- Replace Groq with a locally quantized LLM (e.g., Mistral 7B via Ollama).
- Deploy via K3s (lightweight Kubernetes) directly onto on-premise industrial servers.
- Ensure zero telemetry data leaves the facility's localized network.

---


*Argus Echo demonstrates the integration of Edge AI anomaly detection and RAG-based operator guidance for safety-critical ICS environments.*
