# Argus Echo

**Industrial telemetry anomaly detection and Bolna-powered voice incident decision support.**

Argus Echo trains facility-specific anomaly detectors from normal telemetry, analyzes new sensor data, and provides operator guidance grounded in industrial control system (ICS) security documents.

<a href="https://www.loom.com/share/ae84ea9d2b4740f680059a0cd4ee66a7"><img src="https://cdn.loom.com/sessions/thumbnails/ae84ea9d2b4740f680059a0cd4ee66a7-efaed40b8c2af93e.gif" alt="Click to watch the Argus Echo project walkthrough" width="720"></a>

## Project at a glance

| Area | What this project demonstrates |
| --- | --- |
| AI/ML | LSTM autoencoder for time-series anomaly detection; telemetry preprocessing; rolling variance features; model, scaler, and configuration artifacts |
| MLOps | Per-user model records and artifacts; repeatable training and inference paths; row limits; health probes; persistent storage; a container build workflow |
| DevOps | Docker image; Kubernetes Deployment, Service, namespace, and persistent volume claim; resource requests and limits; readiness and liveness probes |
| Applied AI | Retrieval-augmented ICS guidance using an indexed source corpus, lexical term matching, and Groq generation |
| VoiceOps | Bolna voice agent integration contract, evidence-aware retrieval, low-evidence escalation, incident review, and interaction audit trail |

## User workflow

1. Create an account or log in.
2. Upload normal-operation telemetry and train a facility-specific LSTM autoencoder.
3. Upload telemetry for diagnostics and review reconstruction-error anomalies and history.
4. Ask Echo for an explanation grounded in the indexed ICS security corpus.
5. Ask VoiceOps a direct industrial guidance question through a Bolna voice agent; review cited evidence and confidence or handle an escalation incident.

### How an anomaly answer is grounded

Argus compares each telemetry window with the trained facility baseline and ranks
the contributing sensors by their raw-signal and rolling-variance reconstruction
residuals. It reports the sensor tag, uploaded file/facility label, row index,
recorded values, and model score. A pattern label is descriptive only; it does
not establish a physical fault or cyberattack. The uploaded file and row are the
available location evidence unless verified site-location metadata is present.

For operator action questions, VoiceOps first looks for an exact sensor
procedure. If none is available, it can use relevant CISA incident-response references for
general verification and escalation checks, clearly labels them as general ICS
guidance, and creates a human-review incident. It does not recommend equipment
setpoint changes, resets, bypasses, or control actions without an approved site
procedure. Bolna receives the same current event context and reads the Argus
response, source names, retrieval relevance, and review status during the call.

## Architecture

```mermaid
flowchart LR
    U[Operator] --> UI[Streamlit app]
    UI --> PRE[Telemetry preprocessing<br/>numeric sensors + rolling variance]
    PRE --> TRAIN[LSTM autoencoder training]
    TRAIN --> ART[Model + scaler + config artifacts]
    ART --> DIAG[Diagnostic inference<br/>reconstruction error]
    DIAG --> DB[(SQLite metadata<br/>and anomaly history)]
    DIAG --> RAG[Echo retrieval and guidance]
    CORPUS[ICS security documents] --> IDX[FAISS knowledge index]
    IDX --> RAG
    RAG --> LLM[Groq API]
    RAG --> UI
    B[Bolna voice agent] --> API[FastAPI VoiceOps endpoint]
    UI --> SDK[Bolna Web Call SDK]
    SDK --> B
    API --> RAG
    API --> AUDIT[(SQLite VoiceOps interactions<br/>and human review incidents)]
    UI --> DOCKER[Docker image]
    DOCKER --> K8S[Kubernetes Deployment]
    K8S --> PVC[(Persistent volume)]
    K8S --> SVC[LoadBalancer Service]
```

The application runs as a single Kubernetes replica. SQLite and user artifacts are stored on a persistent volume.

## Pitch materials

- [Five-slide pitch deck](pitch/Argus_Echo_Pitch_Deck_Final.pptx)
- [Timed pitch and walkthrough script](pitch/Argus_Echo_Pitch_Script.md)

## Repository map

```text
.
├── app.py                      Streamlit UI and user workflows
├── argus_model.py              Telemetry preprocessing and LSTM training
├── persistence.py              SQLite users, model metadata, and run history
├── argus_logger.py             Anomaly hint and local logging helpers
├── train_model.py              Command-line training entry point
├── echo/                       Retrieval, indexing, embeddings, and evaluation
├── voiceops_api.py             Authenticated Bolna question and webhook API
├── data/                       ICS security and anomaly-detection references
├── k8s/                        Kubernetes namespace, deployment, and service
├── .github/workflows/          Container build automation
├── Dockerfile                  Runtime container definition
└── requirements*.txt           Development and container dependencies
```

## Run locally

Requires Python 3.10+ and a Groq API key for generated Echo responses. Core model training and diagnostics do not require that key.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:ARGUS_STORAGE_DIR = "ml"
$env:ARGUS_DB_PATH = "ml\argus_echo.db"
$env:GROQ_API_KEY = "your_groq_api_key"
streamlit run app.py
```

Build the local FAISS index from `data/` using the VoiceOps tab. Docker builds the index on first startup if absent. To run the Bolna-facing API locally in a second terminal:

```powershell
$env:ARGUS_VOICEOPS_API_TOKEN = "a-long-random-secret"
$env:ARGUS_BOLNA_WEBHOOK_TOKEN = "a-different-long-random-secret"
uvicorn voiceops_api:app --host 0.0.0.0 --port 8000
```

The VoiceOps API is `POST /api/voiceops/answer` with `Authorization: Bearer <ARGUS_VOICEOPS_API_TOKEN>` and JSON fields `question`, optional `transcript`, optional `interaction_id`, and optional `latest_anomaly` event context. Its JSON response includes `answer`, `spoken_answer`, source excerpts and names, retrieval `confidence`, `confidence_label`, `escalate`, and `model_version`. Configure the Bolna custom function to send the current question and `latest_anomaly` context, then have the agent read `spoken_answer` during the same turn. Do not require `{{execution_id}}` as a tool argument; Bolna's documented function variables do not include it, and web calls may not have a telephony call SID. Configure the agent webhook at `/api/voiceops/bolna-webhook` with `X-Argus-Webhook-Token` set to `ARGUS_BOLNA_WEBHOOK_TOKEN`.

For the live browser call in the VoiceOps tab, set `BOLNA_API_KEY` and `BOLNA_AGENT_ID` as server secrets. The tab requests a short-lived single-use session from `POST /api/bolna-session`; the Bolna key is never sent to the browser. Bolna's Web Call SDK must be enabled for your account. Configure a Bolna **custom function** named `get_documented_operator_guidance` that sends the caller's question and the current sensor context to `POST /api/voiceops/answer`. Have the agent read `spoken_answer` in the same turn and honor `escalate`; never have it invent equipment instructions when the function errors or reports low evidence. Configure the completed-execution webhook in Bolna's Extractions settings to post to `/api/voiceops/bolna-webhook`, adding an `X-Argus-Webhook-Token` header. The endpoint upserts by execution ID because Bolna sends multiple status updates; treat `completed` as final (the `call-disconnected` event can arrive before transcript, duration, and recording data are complete).

The session URL defaults to `http(s)://<same-host>:8000/api/bolna-session` for local Docker, so publish both ports with `-p 8501:8501 -p 8000:8000`. For hosted HTTPS, set `ARGUS_VOICEOPS_API_URL` to the public HTTPS origin that routes `/api/*` to the FastAPI service on port 8000. The UI origin and API route must both be reachable from the browser; do not expose Bolna credentials in the page.

The API credentials are deliberately separate from Bolna's own account API key. Keep all tokens and the Groq key in the host's secret manager; never put them in a public repository or the browser. The Bolna agent and callback must use HTTPS in deployment. Bolna documents source IPs `13.203.39.153`, `13.126.9.249`, and `13.202.133.53` for tool calls and webhooks when firewall allowlisting is used.

The app opens at `http://localhost:8501`. The local storage directory is ignored by Git.

## Build and deploy

### Run with Docker Compose

Copy `.env.example` to `.env`, fill in the Bolna and Groq credentials and the two separate Argus secrets, then start Docker Desktop and run:

```powershell
Copy-Item .env.example .env
# Edit .env locally; do not commit it or share its contents.
docker compose up --build -d
```

Open `http://localhost:8501`. The API health endpoint is `http://localhost:8000/health`. Compose persists the generated FAISS index and SQLite database in the local ignored `ml/` directory. To inspect startup/indexing logs use `docker compose logs -f`; to stop the app use `docker compose down` (the local data directory remains).

For Bolna custom functions and webhooks, local `localhost` is not publicly reachable. Deploy this same Compose service behind public HTTPS before configuring Bolna to call `/api/voiceops/answer` and `/api/voiceops/bolna-webhook`.

GitHub Actions builds the Docker image on pushes to `main` and pull requests. It does not publish to a registry or deploy to a cloud account; deployment remains an explicit operator action.

```powershell
docker build -t argus-echo:local .
docker run --rm -p 8501:8501 -e GROQ_API_KEY=your_groq_api_key argus-echo:local
```

The Kubernetes manifests are in `k8s/`. Configure `GROQ_API_KEY`, `BOLNA_API_KEY`, `BOLNA_AGENT_ID`, `ARGUS_VOICEOPS_API_TOKEN`, and `ARGUS_BOLNA_WEBHOOK_TOKEN` as Kubernetes Secrets before deploying; `k8s/secret.example.yaml` is only a placeholder template. Replace the example `ARGUS_VOICEOPS_API_URL` and `ARGUS_UI_ORIGINS` values in `k8s/deployment.yaml` with the public HTTPS app/API origin. The free deployment target should be selected before public launch: this app stores SQLite and the generated FAISS index locally, so a host with ephemeral storage will lose audit records and index files on restart unless the index is rebuilt and durable database storage is configured.

## Model and data notes

- Timestamp/date columns and target labels such as `Attack` are excluded from model features. When a recognized binary attack label is included in training data, labelled attack rows are removed so the baseline learns normal operation only.
- Rolling variance features are added per sensor.
- The final 10% of normal training telemetry is held out to calibrate a model-specific reconstruction-error threshold at the 99.5th percentile. This sets a 0.5% target exceedance rate on the held-out normal calibration slice; it is not a guarantee of the same false-positive rate on future data.
- Diagnostics align inputs to the saved model feature order and report precision, recall, F1, and false-positive rate when the CSV includes both normal and attack labels.
- Training and diagnostic files must include the same sensor features. Legacy models without a calibrated threshold must be retrained before diagnostics.
- Training and diagnostic caps default to 20,000 rows and are sampled evenly across the full CSV in source order, so later operating modes and test intervals are represented; event rows retain their original CSV row numbers. Raise the caps for short-lived events that may fall between sampled rows.
- User identity and operational metadata use SQLite; trained models and generated data live on local or mounted storage.

## Technology

Python · pandas · NumPy · TensorFlow/Keras · scikit-learn · Streamlit · SQLite · FAISS · LangChain · Groq · Docker · Kubernetes · Google Kubernetes Engine
