"""AWS Lambda entrypoint — wraps the same FastAPI app (`api.main.app`) with
Mangum so it can run behind a Lambda Function URL. Not used locally or in
Docker Compose (those run `api.main.app` directly via uvicorn); see
Dockerfile.lambda.
"""

from mangum import Mangum

from .main import app

handler = Mangum(app)
