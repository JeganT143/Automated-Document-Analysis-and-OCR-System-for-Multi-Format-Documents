// Where the static frontend finds the API. Overwritten at deploy time (CI
// rewrites this one line to the deployed Lambda Function URL) — see
// .github/workflows/deploy-aws.yml. Left as localhost for local dev.
window.API_BASE_URL = "http://localhost:8010";
