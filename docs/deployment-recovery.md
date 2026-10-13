# Deployment recovery runbook

The previously documented public endpoint was `34.121.158.211`, but the repository cannot confirm whether that address is still assigned to a live GKE service. A Kubernetes LoadBalancer address may change if the service or cluster was recreated. Use the current Service ingress as the source of truth.

## 1. Check cloud access and active service

Run these commands from a machine with Google Cloud SDK, `kubectl`, and access to the `argus-echo` project:

```powershell
gcloud auth list
gcloud config set project argus-echo
gcloud container clusters list --region us-central1
gcloud container clusters get-credentials argus-echo-cluster --region us-central1 --project argus-echo
kubectl -n argus-echo get deployment,pods,svc,pvc -o wide
```

If the cluster is absent, stopped, or inaccessible, resolve that cloud-account/project state first. Do not assume the old IP is reusable.

## 2. Inspect health and current public address

```powershell
kubectl -n argus-echo get svc argus-echo -o jsonpath="{.status.loadBalancer.ingress[0].ip}"
kubectl -n argus-echo rollout status deployment/argus-echo --timeout=300s
kubectl -n argus-echo logs deployment/argus-echo --tail=150
```

Copy the returned ingress IP and check both the page and Streamlit health endpoint:

```powershell
curl.exe -i http://<CURRENT_INGRESS_IP>/
curl.exe -i http://<CURRENT_INGRESS_IP>/_stcore/health
```

Expected health response is HTTP 200. If the service has no ingress IP, inspect `kubectl -n argus-echo describe svc argus-echo` and the GKE load balancer events. If pods are not ready, inspect `kubectl -n argus-echo describe pod <POD_NAME>` and the logs before restarting anything.

## 3. Check application configuration

The deployment expects the `argus-echo-secrets` Secret with a `GROQ_API_KEY` key. The committed `k8s/secret.example.yaml` contains a placeholder and should not be applied to a live cluster.

Create or update the Secret without putting the API key into Git:

```powershell
$env:GROQ_API_KEY = "your_real_groq_api_key"
kubectl -n argus-echo create secret generic argus-echo-secrets `
  --from-literal=GROQ_API_KEY=$env:GROQ_API_KEY `
  --dry-run=client -o yaml | kubectl apply -f -
Remove-Item Env:GROQ_API_KEY
```

Echo generation needs this key. The UI, model training, and diagnostics can start without it, but Echo requests will fail until a valid key is configured.

## 4. Build and deploy a new image

GitHub Actions currently validates the Docker image build but does not push or deploy it. From a machine authenticated to Google Cloud and Artifact Registry:

```powershell
$image = "us-central1-docker.pkg.dev/argus-echo/argus-echo/argus-echo:$(git rev-parse --short HEAD)"
gcloud auth configure-docker us-central1-docker.pkg.dev
docker build -t $image .
docker push $image
kubectl -n argus-echo set image deployment/argus-echo argus-echo=$image
kubectl -n argus-echo rollout status deployment/argus-echo --timeout=300s
```

If the new release is unhealthy, run this rollback separately:

```powershell
kubectl -n argus-echo rollout undo deployment/argus-echo
```

Once a commit-tag deployment is in place, update `k8s/deployment.yaml` to the chosen immutable tag or digest for reproducible future applies.

## 5. Publish the working link

After health checks pass, use the current Service ingress address as the temporary demo link. For a stable portfolio URL, reserve a static external IP, point a domain's DNS record to it, and configure TLS termination. Update the README only with a URL that has been checked from outside the cluster.

## Local constraint during this repair

This checkout environment cannot reach the external endpoint, and Google Cloud/Kubernetes CLIs are not available here. Therefore no cloud resource, current ingress IP, or restored public link could be verified from this workspace.
