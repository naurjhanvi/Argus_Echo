# Architecture and MLOps

## Request and model lifecycle

1. A user uploads a CSV through the Streamlit application.
2. `argus_model.py` removes timestamp and label-like fields, keeps numeric telemetry, and adds rolling variance features.
3. Training scales the features, windows the time series, and fits an LSTM autoencoder on normal-operation data.
4. The application stores the Keras model, scaler, and feature configuration with the user's model record.
5. Diagnostics preprocess telemetry using the same feature pipeline, load the chosen artifacts, and calculate reconstruction errors.
6. Anomaly records and run metadata are written to SQLite. Echo retrieves context from the indexed ICS documents and can ask Groq to generate operator guidance.

## Runtime components

| Component | Responsibility | Current implementation |
| --- | --- | --- |
| Streamlit | Authentication, training, diagnostics, visualizations, and Echo UI | One Python process in one pod |
| ML pipeline | Preprocessing, sequence creation, training, and scoring | TensorFlow/Keras LSTM autoencoder |
| Metadata | Accounts, models, runs, anomalies, and generated answers | SQLite database on the persistent volume |
| Artifacts | Model, scaler, config, uploaded telemetry, and vector index | Files under the mounted `/app/ml` storage path |
| Knowledge retrieval | Index and search ICS documents | FAISS with local embeddings |
| Text generation | Compose grounded operator-facing guidance | External Groq API; `GROQ_API_KEY` is supplied as a Kubernetes Secret |
| Runtime | Package and run Streamlit | Docker image, port 8501 |
| Orchestration | Schedule the app and expose it | GKE Deployment and LoadBalancer Service |

## MLOps controls present

- Model artifacts and feature metadata are saved together for later inference.
- The model record is associated with the account that trained it.
- Training and diagnostics cap row counts to bound runtime memory use.
- Kubernetes declares CPU, memory, and ephemeral-storage requests and limits.
- Kubernetes readiness/liveness probes and the image health check use Streamlit's health endpoint.
- SQLite and file artifacts persist on a PVC rather than the container's writable layer.
- Container image builds run through GitHub Actions for pull requests and `main` updates.

## Known operational gaps

- A single pod, SQLite, and a `ReadWriteOnce` PVC do not support safe horizontal scaling.
- Model threshold and quality are not yet tracked as a versioned evaluation report; the current threshold is a fixed application constant.
- Training runs in the web application process. Long jobs can tie up a session and compete with inference.
- There is no model registry, immutable release promotion, drift monitoring, or automated rollback.
- The current Kubernetes image reference uses `:latest`; production should deploy an immutable commit tag or digest.
- The previous public IP in the README is stale or at least unverified; a reserved address and domain with TLS are preferable.
- The application should receive authentication hardening, rate limits, and a production database/object store before broader access.

## Suggested next MLOps iteration

1. Add a reproducible evaluation step with precision/recall at an agreed operating threshold, false-positive review, and a versioned report.
2. Move training and diagnostics into background workers with explicit job status and bounded concurrency.
3. Store metadata in Postgres and artifacts in object storage; keep artifact URIs and content hashes with model records.
4. Introduce model versions and a promotion gate so only evaluated artifacts can be deployed.
5. Deploy immutable image digests, configure a reserved IP and TLS domain, and add an external uptime check.
6. Capture request latency, training duration, failures, resource use, and data/model drift indicators.

These are roadmap items, not claims about functionality already delivered.
