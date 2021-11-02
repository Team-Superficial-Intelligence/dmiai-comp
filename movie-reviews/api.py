import uvicorn
from fastapi import FastAPI
from starlette.responses import HTMLResponse

import middleware.cors
import middleware.logging
from dtos.requests import PredictRequest
from dtos.responses import PredictResponse

from settings import Settings, load_env
from static.render import render
from utilities.utilities import get_uptime
import random
from ml.emily import Emily

emily = Emily()

load_env()

# --- Welcome to your Emily API! --- #
# See the README for guides on how to test it.

# Your API endpoints under http://yourdomain/api/...
# are accessible from any origin by default.
# Make sure to restrict access below to origins you
# trust before deploying your API to production.

app = FastAPI()
settings = Settings()

middleware.logging.setup(app)
middleware.cors.setup(app)


@app.post('/api/predict', response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:

    # You receive all reviews as plaintext in the request.
    # Return a list of predicted ratings between 1-5 (inclusive).
    # You must return the same number of ratings as there are reviews, and each
    # rating will be associated with the review at the same index in the request list.

    ratings = [random.uniform(0.5, 5.0) for review in request.reviews]
    return PredictResponse(ratings=ratings)


@app.get('/superfuntime')
def superfuntime():
    tok = emily.superfuntime()
    return HTMLResponse(str(tok))


@app.get('/api')
def hello():
    return {
        "uptime": get_uptime(),
        "service": settings.COMPOSE_PROJECT_NAME,
    }


@app.get('/')
def index():
    return HTMLResponse(
        render('static/index.html',
               host=settings.HOST_IP,
               port=settings.CONTAINER_PORT))


if __name__ == '__main__':

    uvicorn.run('api:app', host=settings.HOST_IP, port=settings.CONTAINER_PORT)
