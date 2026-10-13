# Portfolio and interview guide

## One-line introduction

Argus Echo is an end-to-end industrial MLOps demo that trains user-specific time-series anomaly detectors, serves diagnostics through a web app, and packages the system for Kubernetes.

## Role-specific story

### AI/ML Engineer

- Explain the normal-operation training assumption and how reconstruction error becomes an anomaly score.
- Walk through numeric-feature selection, rolling variance features, scaling, sliding windows, and the LSTM autoencoder.
- Discuss how the fixed anomaly threshold should be calibrated and evaluated against labeled facility data.
- Explain the limits of the current model: facility-specific feature layouts, threshold sensitivity, and the need for false-positive/recall measurements.

### MLOps Engineer

- Walk through the path from uploaded telemetry to a model, scaler, feature configuration, inference, and persisted diagnostic records.
- Explain how row caps, Kubernetes resource limits, health probes, and persistent storage shape runtime behavior.
- Discuss model versioning, repeatable evaluation, artifact hashes, background training jobs, and promotion gates as next steps.
- Be clear that current CI builds the container; it does not publish an image or deploy it to GKE.

### DevOps Engineer

- Describe the Docker runtime, GKE Deployment and Service, persistent volume claim, secret injection, and health checks.
- Demonstrate how to inspect pods, rollout status, logs, and LoadBalancer ingress using the recovery runbook.
- Explain why image tags should be immutable, why secrets should stay outside Git, and why the public IP should be reserved behind a TLS domain.
- Be clear that the live IP previously listed in the README is unverified and that current cloud resources still need an operator-side check.

## Resume bullets

Use these as starting points and keep claims aligned with what you can demonstrate:

- Built an industrial telemetry anomaly-detection application using a TensorFlow LSTM autoencoder, sliding-window sequences, scaled sensor values, and rolling-variance features.
- Implemented a multi-user Streamlit workflow for model training, artifact persistence, diagnostic inference, anomaly history, and retrieval-grounded ICS guidance.
- Containerized the application and authored GKE manifests with resource bounds, persistent storage, secret injection, and readiness/liveness probes.
- Added pull-request and main-branch container build automation with GitHub Actions; cloud image publishing and deployment remain manual.

Add measured outcomes only after running a documented evaluation on a known dataset. Do not invent model accuracy, uptime, latency, cost savings, or deployment status.

## Demo sequence

1. Start with the architecture diagram and identify the model, data, app, and deployment boundaries.
2. Show training on normal telemetry and explain the saved model/scaler/config bundle.
3. Run diagnostics on telemetry with known anomalies and discuss threshold tradeoffs.
4. Show the Echo retrieval flow and the source knowledge documents.
5. Show the Docker build workflow and Kubernetes health, persistence, and rollout configuration.
6. Close with the known scaling and model-monitoring gaps and the next MLOps iteration.

## Highest-value next improvements

1. Establish a verified public URL with TLS and a simple externally monitored health check.
2. Add a reproducible model evaluation report with labeled examples and operating-threshold analysis.
3. Replace `latest` releases with immutable image digests and a gated deployment workflow.
4. Move SQLite and file artifacts to managed database and object storage, then run training as asynchronous jobs.
5. Add structured logs, latency/error/resource dashboards, and data/model drift checks.
