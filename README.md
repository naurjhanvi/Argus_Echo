# Argus Echo

<a href="https://www.loom.com/share/ae84ea9d2b4740f680059a0cd4ee66a7"><img src="https://cdn.loom.com/sessions/thumbnails/ae84ea9d2b4740f680059a0cd4ee66a7-efaed40b8c2af93e.gif" alt="Watch the Argus Echo walkthrough" width="720"></a>

[Watch the video on Loom](https://www.loom.com/share/ae84ea9d2b4740f680059a0cd4ee66a7)

Universal Edge AI diagnostics for industrial telemetry.

Argus Echo is a Streamlit application that lets a user train a facility-specific anomaly detection model from normal telemetry, run diagnostics on new telemetry, and ask Echo for operator guidance grounded in an ICS security knowledge base.

This repository is an extension of the original Project Argus work. Argus Echo turns that core ML/anomaly-detection foundation into a deployed multi-user diagnostic app with accounts, saved model artifacts, diagnostic history, Echo guidance, Docker packaging, and GKE deployment files.

To understand the core ML approach, model motivation, anomaly detection logic, and original research direction, refer to Project Argus:

https://github.com/naurjhanvi/project-argus

Live app:

http://34.121.158.211


## What The App Does

Argus Echo has three main flows:

1. Train Model

   Upload a normal-operation telemetry CSV. Argus preprocesses numeric sensor columns, adds rolling variance features, trains an LSTM autoencoder, and saves the trained model for that user account.

2. Run Diagnostics

   Upload a telemetry CSV to analyze. Argus loads the selected trained model, checks the feature shape, computes reconstruction error, highlights anomalous windows, stores the run, and shows charts/history in the dashboard.

3. Echo Guidance

   Build a knowledge base from the documents in `data/`, then ask Echo for operator guidance based on the latest anomaly. Echo uses retrieval augmented generation with Groq to produce ICS-focused response text.

## Current Production URL

The current deployed app is available at:

```text
http://34.121.158.211
```

Anyone with the URL can open the app. They still need to create an account or log in inside the app before training models or running diagnostics.

## Current Deployment

The app is deployed on Google Cloud using:

- Google Kubernetes Engine: `argus-echo-cluster`
- GCP project: `argus-echo`
- Region: `us-central1`
- Artifact Registry image:
  `us-central1-docker.pkg.dev/argus-echo/argus-echo/argus-echo:latest`
- Kubernetes namespace: `argus-echo`
- Kubernetes deployment: `argus-echo`
- Kubernetes service: `argus-echo`
- Service type: `LoadBalancer`
- Public IP: `34.121.158.211`

The container serves Streamlit on port `8501`. The Kubernetes service exposes it publicly on port `80`.

## Data Storage

Current production storage is file-based and SQLite-based.

Inside the GKE pod:

```text
/app/ml
```

This path is mounted from the Kubernetes persistent volume claim:

```text
argus-echo-storage
```

The database is:

```text
/app/ml/argus_echo.db
```

The SQLite database stores:

- user accounts
- password salts and password hashes
- model metadata
- diagnostic run metadata
- anomaly records
- Echo answers

Model and diagnostic artifacts are stored under:

```text
/app/ml/users/<user_id>/models/<model_id>/
/app/ml/users/<user_id>/diagnostics/<run_id>/
```

Echo vector indexes are stored under:

```text
/app/ml/vectorstore/echo/
```

### Important Storage Limitation

This is acceptable for a demo or small controlled test, but it is not the right final architecture for many users.

The current app runs one pod with SQLite on a persistent volume. If many users train models or run diagnostics at the same time, the app can become slow or hit SQLite write contention. Scaling to multiple pods would also be awkward because the PVC is configured as `ReadWriteOnce`.

For a real multi-user production version, move to:

- Supabase Postgres for users, model metadata, diagnostic runs, anomalies, and Echo answers
- Supabase Storage or Google Cloud Storage for trained model files, scalers, configs, diagnostic outputs, and uploaded documents
- background jobs for training and diagnostics instead of doing heavy work inside the Streamlit request/session process

## How Users Use The App

1. Open the live URL:

   ```text
   http://34.121.158.211
   ```

2. Create an account from the sidebar.

3. Open the `Train Model` tab.

4. Upload a CSV containing normal-operation telemetry.

5. Choose:

   - model name
   - training epochs
   - batch size

6. Click `Train Model`.

7. Wait for training to finish. The app saves the model under the logged-in account.

8. Open the `Run Diagnostics` tab.

9. Select the trained model.

10. Upload a telemetry CSV for analysis.

11. Select sensors to visualize.

12. Click `Run Diagnostics`.

13. Review anomaly scores, charts, anomaly records, and diagnostic history.

14. Open `Echo Guidance`.

15. If the Echo knowledge base is not built yet, click `Build Knowledge Base From data/`.

16. Ask Echo for guidance on the latest anomaly.

## CSV Expectations

The training and diagnostic files should use the same telemetry shape.

The app:

- drops timestamp/date/time/label-like columns
- keeps numeric columns
- adds rolling variance features for every numeric sensor
- expects the diagnostic CSV to produce the same feature count as the trained model

Example:

```text
87 raw numeric sensors -> 174 model features
```

If a diagnostic file has a different sensor layout, train a new model for that layout.

## Current Runtime Limits

The deployed app currently caps rows to keep memory and CPU usage reasonable inside the GKE pod:

```text
ARGUS_MAX_TRAINING_ROWS=20000
ARGUS_MAX_DIAGNOSTIC_ROWS=20000
```

Large CSV files can still be uploaded, but the model flow samples/caps rows before training or diagnostics.

## Local Development

### Prerequisites

- Python 3.10+
- Docker Desktop, only if building or testing the container locally
- Google Cloud SDK, only if deploying to GCP
- kubectl, only if deploying to GKE
- Groq API key for Echo guidance

### Install Python Dependencies

```powershell
cd C:\Users\<username>\projects\Argus_Echo
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### Run Locally

```powershell
streamlit run app.py
```

By default, local data is stored under:

```text
ml/
```

You can override storage paths with environment variables:

```powershell
$env:ARGUS_STORAGE_DIR = "ml"
$env:ARGUS_DB_PATH = "ml\argus_echo.db"
$env:GROQ_API_KEY = "your_groq_api_key"
$env:GROQ_MODEL = "openai/gpt-oss-120b"
streamlit run app.py
```

## Docker Build

Build the production image:

```powershell
docker build -t us-central1-docker.pkg.dev/argus-echo/argus-echo/argus-echo:latest .
```

Push it:

```powershell
docker push us-central1-docker.pkg.dev/argus-echo/argus-echo/argus-echo:latest
```

Docker Desktop is only needed while building, tagging, running, or pushing images from your machine. Once the image is pushed and Kubernetes is running it in GKE, closing Docker Desktop does not stop the public website.

## Deploy To GKE

Set the project:

```powershell
gcloud config set project argus-echo
```

Get cluster credentials:

```powershell
gcloud container clusters get-credentials argus-echo-cluster --region us-central1 --project argus-echo
```

Apply Kubernetes manifests:

```powershell
kubectl apply -f k8s\namespace.yaml
kubectl apply -f k8s\secret.example.yaml
kubectl apply -f k8s\deployment.yaml
kubectl apply -f k8s\service.yaml
```

Restart the deployment after pushing a new `latest` image:

```powershell
kubectl -n argus-echo rollout restart deployment/argus-echo
kubectl -n argus-echo rollout status deployment/argus-echo --timeout=300s
```

Check pods:

```powershell
kubectl -n argus-echo get pods -o wide
```

Check the public service:

```powershell
kubectl -n argus-echo get svc
```

Check logs:

```powershell
kubectl -n argus-echo logs deployment/argus-echo --tail=100
```

Health check:

```powershell
curl http://34.121.158.211/_stcore/health
```

## Project Structure

```text
argus_echo/
|-- app.py                    Streamlit UI and app workflow
|-- argus_model.py            Telemetry preprocessing, LSTM model, training helpers
|-- persistence.py            SQLite users, models, diagnostics, anomalies
|-- argus_logger.py           Attack hint helper logic
|-- Dockerfile                Production container image
|-- requirements.txt          Python dependencies
|-- data/                     ICS/security corpus documents for Echo
|-- echo/
|   |-- embeddings.py         Local deterministic embeddings for FAISS
|   |-- ingest.py             Builds Echo FAISS knowledge base
|   |-- rag.py                Echo retrieval and Groq generation
|   |-- query_builder.py      Converts anomaly records into Echo prompts
|-- k8s/
|   |-- namespace.yaml        Kubernetes namespace
|   |-- secret.example.yaml   Secret manifest template/current secret manifest
|   |-- deployment.yaml       GKE deployment and PVC
|   |-- service.yaml          Public LoadBalancer service
```

## Technology Used

- Python
- Streamlit
- TensorFlow/Keras
- scikit-learn
- pandas
- NumPy
- Plotly
- SQLite
- FAISS
- LangChain community vector store integration
- Groq GPT-OSS 120B for Echo guidance (override with `GROQ_MODEL`)
- Docker
- Google Artifact Registry
- Google Kubernetes Engine
- Kubernetes LoadBalancer service

## Operational Notes

- The app currently runs as one GKE replica.
- The app stores state in `/app/ml` through a persistent volume claim.
- The public URL remains live when the local laptop is closed, as long as the GKE cluster, pod, service, and billing remain active.
- New deployments require Docker Desktop only for building/pushing from the local machine.
- Supabase is the recommended next database/storage step before inviting many concurrent users.
