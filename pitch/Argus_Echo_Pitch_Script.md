# Argus Echo: pitch and walkthrough script

**Target runtime: 4 minutes 30 seconds**
**Format:** narration over the five-slide deck, with a short live walkthrough of the app.

## Before recording

- Train and diagnose a fresh run first. The previously saved model and diagnostic results were cleared for this walkthrough.
- Have the BATADAL normal-operation training CSV and a separate telemetry CSV ready. The current `data` folder has the C-Town `.INP` network file and reference documents, but no BATADAL telemetry CSV.
- Open the app at `http://localhost:8501/` and confirm the training, diagnostic, Echo, and VoiceOps screens load.
- Try the Bolna call once before recording. A live Bolna agent calling Argus needs a publicly reachable HTTPS API URL. If that is not configured, demonstrate the VoiceOps text question and say the live callback needs a public HTTPS endpoint.
- Keep the voice example within the BATADAL simulation. Do not describe it as advice for real equipment.

## Timed narration

### 0:00–0:20 | Slide 1 — Argus Echo

“Industrial systems produce a lot of sensor data. When a signal changes, a red flag alone does not tell an operator what it means. Argus Echo combines anomaly detection, document search, and a voice assistant to help make that signal easier to review.”

### 0:20–0:50 | Slide 2 — The operator’s problem

“Argus learns a baseline from normal telemetry and flags readings that differ. The score describes a deviation. It does not prove a fault or a cyberattack. This demo uses the simulated BATADAL C-Town water network. Its public references explain the benchmark, not how to operate real equipment.”

### 0:50–1:15 | Slide 3 — How Argus works

“The model is an LSTM autoencoder. Diagnostics compare new telemetry with the learned baseline and show the strongest sensor residuals. Echo searches indexed reference text for matching terms, then Groq drafts a sourced explanation. The retrieval score reflects term overlap, not answer correctness. Bolna adds voice, passing the question and event context to Argus and speaking its response.”

### 1:15–3:55 | Slide 4 — Live walkthrough

**Show Train Model.**

“I’ll train on normal BATADAL telemetry. Argus saves the model and its scoring settings.”

**Show Run Diagnostics and the prepared test telemetry.**

“Now I’ll check a separate telemetry file. The result shows the sensor, row, reconstruction error, and strongest contributing signals. It measures deviation from the baseline, not a confirmed diagnosis.”

**Show Echo Guidance and the retrieved sources.**

“I’ll ask Echo, ‘What does S V two represent in the BATADAL benchmark?’ It searches the indexed references, explains the simulated sensor tag, and shows its source.”

**Open VoiceOps and start the Bolna session or use the text question if live handoff is not configured.**

“Now I’ll ask the same question through Bolna. VoiceOps sends the question and event context to Argus, then reads the answer. I’ll also ask what action to take on the simulated valve. Since there is no approved procedure for it, Argus should state that limit and escalate for human review rather than invent instructions.”

### 3:55–4:25 | Slide 5 — Safe, useful decision support

“Argus connects a learned telemetry baseline with retrieved evidence and voice. It helps operators inspect a deviation while keeping uncertainty visible. This prototype uses simulated data. It does not control equipment or replace site procedures. The next step is evaluation with labeled data and approved procedures for a defined facility. Thank you.”

## Recording notes

- Keep the browser walkthrough brief. The slide narration plus live actions should stay below five minutes.
- The demo needs a fresh trained model because the saved model and results were intentionally cleared.
- If Groq, FAISS, or Bolna is unavailable during recording, do not imply a response came from that service. Show the status honestly and continue with the prepared screenshots or explain the dependency briefly.
- Avoid promising that a deviation is an attack, that a physical location is known, or that a human operator will respond within a specific time.
