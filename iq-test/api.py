import uvicorn
from fastapi import FastAPI
from starlette.responses import HTMLResponse

import middleware.cors
import middleware.logging
from dtos.requests import PredictRequest
from dtos.responses import PredictResponse

from shit_checker import full_check

from settings import Settings, load_env
from static.render import render
from utilities.utilities import get_uptime
import random

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

    print("\n\n\n====================\n\n\n")

    # You receive image in base64 encoding.
    image = request.image_base64
    choices = request.image_choices_base64

    # Process the first two images, and predict the next correct image
    # from the list of image choices

    # Dummy prediction - chooses a random image from the list of choices
    next_image_index = full_check.check_shit(image, choices)

    return PredictResponse(next_image_index=next_image_index)


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
