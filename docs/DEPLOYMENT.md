# Deploying to Google Cloud Run

Two Cloud Run services, both free-tier-eligible (scale to zero — you pay
nothing while nobody is using the demo):

| Service | What | URL |
|---|---|---|
| `docai-api` | FastAPI backend (OCR + LLM layer) | its default `*.run.app` URL — no custom domain needed |
| `docai-web` | Streamlit UI, calls the api | mapped to **`docai.jegant.dev`** |

Only the UI gets a pretty URL. The API is only ever called by the UI's
server-side code (not from a visitor's browser), so it doesn't need one —
that halves the domain-verification/DNS work. Swap `docai` for whatever
subdomain you'd rather use; it's a single value everywhere below.

Cloud Run's free tier: 2,000,000 requests/month, 360,000 GiB-seconds and
180,000 vCPU-seconds of compute — enough for a portfolio demo indefinitely,
as long as nothing is set to `min-instances > 0` (see the cost note in
§6). You still need billing enabled on the project; you just won't be
charged at this scale.

---

## 0. Prerequisites

- A Google Cloud account with billing enabled (enabling billing is required
  even to stay inside the free tier).
- The [`gcloud` CLI](https://cloud.google.com/sdk/docs/install), authenticated:
  `gcloud auth login`.
- Docker (already installed, used earlier in this project for local testing).
- Ownership of `jegant.dev` with access to its DNS records.
- This repo pushed to GitHub.
- Your OpenRouter API key, ready to paste into Secret Manager (not into any
  file in the repo).

---

## 1. Create/select a project and enable APIs

```bash
gcloud projects create docai-portfolio --name="Document OCR + AI Pipeline"
gcloud config set project docai-portfolio

gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  iamcredentials.googleapis.com \
  secretmanager.googleapis.com
```

Pick a region that supports Cloud Run domain mappings —
`us-central1` is the safest choice. Set it once so later commands can reuse it:

```bash
REGION=us-central1
PROJECT_ID=$(gcloud config get-value project)
gcloud config set run/region "$REGION"
```

## 2. Artifact Registry (image storage)

```bash
gcloud artifacts repositories create docai \
  --repository-format=docker \
  --location="$REGION" \
  --description="Document OCR + AI Pipeline images"

gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet
```

## 3. Store the OpenRouter key in Secret Manager

Never put the key in a Dockerfile, a committed `.env`, or a `--set-env-vars`
flag (those end up in shell history and `gcloud` audit logs in plaintext).
Secret Manager keeps it out of both:

```bash
printf '%s' 'YOUR_OPENROUTER_KEY' | gcloud secrets create openrouter-api-key \
  --data-file=- --replication-policy=automatic
```

(To rotate it later: `printf '%s' 'NEW_KEY' | gcloud secrets versions add openrouter-api-key --data-file=-`.)

## 4. First deploy (manual, to verify everything works end-to-end)

Build and push both images:

```bash
docker build -f Dockerfile.api -t "${REGION}-docker.pkg.dev/${PROJECT_ID}/docai/api:v1" .
docker push "${REGION}-docker.pkg.dev/${PROJECT_ID}/docai/api:v1"

docker build -f Dockerfile.web -t "${REGION}-docker.pkg.dev/${PROJECT_ID}/docai/web:v1" .
docker push "${REGION}-docker.pkg.dev/${PROJECT_ID}/docai/web:v1"
```

Deploy the api service first — it doesn't depend on anything else:

```bash
gcloud run deploy docai-api \
  --image "${REGION}-docker.pkg.dev/${PROJECT_ID}/docai/api:v1" \
  --allow-unauthenticated \
  --min-instances=0 --max-instances=3 \
  --memory=1Gi --cpu=1 \
  --session-affinity \
  --set-secrets=OPENROUTER_API_KEY=openrouter-api-key:latest
```

`--session-affinity` is a best-effort hint so a given browser session tends
to land on the same instance — it makes the in-memory, session-scoped
document store (see `src/llm/search.py`) work more often across a single
visitor's requests. It's best-effort, not a guarantee (Cloud Run can still
reassign on scale events) — that's a deliberate, documented tradeoff, not a
bug; see the README's Limitations section.

Grab its URL and deploy the web service pointed at it:

```bash
API_URL=$(gcloud run services describe docai-api --format='value(status.url)')
echo "$API_URL"

gcloud run deploy docai-web \
  --image "${REGION}-docker.pkg.dev/${PROJECT_ID}/docai/web:v1" \
  --allow-unauthenticated \
  --min-instances=0 --max-instances=3 \
  --memory=512Mi --cpu=1 \
  --set-env-vars="API_BASE_URL=${API_URL}"
```

Open the web service's `*.run.app` URL (from the deploy output, or
`gcloud run services describe docai-web --format='value(status.url)'`) and
run a document through it before touching DNS — confirm the OCR pipeline,
LLM extraction (if the key is valid), Q&A and search all work against the
real deployed services.

## 5. Map the custom domain

Domain mapping needs the domain verified under the **same Google account**
running `gcloud` (Search Console verification):

1. Go to [Google Search Console](https://search.google.com/search-console),
   add `jegant.dev` as a Domain property, and verify it (Search Console gives
   you a DNS TXT record to add — do that at your DNS provider first).
2. Once verified:

```bash
gcloud beta run domain-mappings create \
  --service=docai-web \
  --domain=docai.jegant.dev \
  --region="$REGION"
```

3. The command prints the DNS records to add (typically a CNAME to
   `ghs.googlehosted.com`). Add exactly that record at your DNS provider for
   `jegant.dev`.
4. Cloud Run provisions a managed TLS certificate automatically once DNS
   resolves — this can take anywhere from a few minutes to a few hours.
   Check status with:

```bash
gcloud beta run domain-mappings describe --domain=docai.jegant.dev --region="$REGION"
```

Note: Cloud Run domain mappings are a **Preview** feature (not GA) in some
regions — `us-central1` has long-standing support and is the recommended
region for exactly this reason. If it's ever unavailable in your region, the
fallback is a Global External Application Load Balancer with a Serverless
NEG, which is GA but adds a small (~$18+/month) forwarding-rule cost — not
worth it for a portfolio demo, so this guide doesn't use it.

## 6. Cost controls (read this before you forget about it)

- Leave `--min-instances=0` on both services (the deploy commands above
  already do). That's what makes this free at low traffic — an idle service
  costs nothing. Setting `min-instances=1` to avoid cold starts would burn
  through the free vCPU-second allotment in days, not months.
- `--max-instances=3` caps a runaway/abusive traffic spike from scaling
  unboundedly and costing money.
- Check current usage any time: **Cloud Console → Billing → Reports**, or
  `gcloud billing accounts list` / the Cloud Run service's Metrics tab.
- To tear everything down later:
  ```bash
  gcloud run services delete docai-api docai-web --region="$REGION" --quiet
  gcloud artifacts repositories delete docai --location="$REGION" --quiet
  gcloud secrets delete openrouter-api-key --quiet
  ```

## 7. Automatic deploys via GitHub Actions (Workload Identity Federation)

This wires `.github/workflows/deploy.yml` (already in the repo) to deploy on
every push to `main` — no long-lived service-account key ever leaves Google
Cloud; GitHub's own OIDC token is exchanged for short-lived credentials at
run time.

**7.1 Create a dedicated deploy service account** (least-privilege — it can
deploy Cloud Run services and push images, nothing more):

```bash
gcloud iam service-accounts create docai-deployer \
  --display-name="Document OCR CI/CD deployer"

DEPLOYER="docai-deployer@${PROJECT_ID}.iam.gserviceaccount.com"

for role in roles/run.admin roles/artifactregistry.writer roles/iam.serviceAccountUser; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:${DEPLOYER}" --role="$role"
done
```

**7.2 Create the Workload Identity Pool + Provider**, scoped to your repo
specifically (replace `JeganT143/REPO_NAME` with your actual `owner/repo`):

```bash
gcloud iam workload-identity-pools create github-pool \
  --location=global --display-name="GitHub Actions pool"

gcloud iam workload-identity-pools providers create-oidc github-provider \
  --location=global --workload-identity-pool=github-pool \
  --display-name="GitHub OIDC provider" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
  --attribute-condition="assertion.repository=='JeganT143/REPO_NAME'" \
  --issuer-uri="https://token.actions.githubusercontent.com"

WIF_POOL_ID=$(gcloud iam workload-identity-pools describe github-pool --location=global --format='value(name)')

gcloud iam service-accounts add-iam-policy-binding "$DEPLOYER" \
  --role=roles/iam.workloadIdentityUser \
  --member="principalSet://iam.googleapis.com/${WIF_POOL_ID}/attribute.repository/JeganT143/REPO_NAME"
```

**7.3 Get the full provider resource name** (this is the `GCP_WIF_PROVIDER`
secret value):

```bash
gcloud iam workload-identity-pools providers describe github-provider \
  --location=global --workload-identity-pool=github-pool --format='value(name)'
```

**7.4 In the GitHub repo → Settings → Secrets and variables → Actions:**

- Secrets: `GCP_WIF_PROVIDER` (output of 7.3), `GCP_DEPLOY_SERVICE_ACCOUNT`
  (the `docai-deployer@...` email from 7.1).
- Variables: `GCP_PROJECT_ID`, `GCP_REGION` (`us-central1`).

**7.5 Grant the runtime service accounts secret access** — deploys run as
`docai-deployer`, but the *running* Cloud Run services use the Compute
Engine default service account unless you set `--service-account` explicitly.
Grant it Secret Manager access once:

```bash
RUNTIME_SA=$(gcloud iam service-accounts list --filter="displayName:'Compute Engine default service account'" --format='value(email)')
gcloud secrets add-iam-policy-binding openrouter-api-key \
  --member="serviceAccount:${RUNTIME_SA}" --role=roles/secretmanager.secretAccessor
```

Push to `main` and watch the **Actions** tab — `deploy.yml` builds both
images, pushes them to Artifact Registry, and redeploys both services.

## 8. Verify

```bash
curl -s "$(gcloud run services describe docai-api --region="$REGION" --format='value(status.url)')/healthz"
curl -sI https://docai.jegant.dev | head -1
```

Then open `https://docai.jegant.dev` in a browser and run a document
through the full flow (upload → OCR → LLM extraction → ask → search) against
the live deployment.
