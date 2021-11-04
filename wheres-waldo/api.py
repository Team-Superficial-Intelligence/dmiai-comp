import walbrute

import uvicorn
from fastapi import FastAPI, File
from starlette.responses import HTMLResponse

import middleware.cors
import middleware.logging
from dtos.requests import PredictRequest
from dtos.responses import PredictResponse

from settings import Settings, load_env
from static.render import render
from utilities.utilities import get_uptime

from PIL import Image
import io

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


@app.post("/api/predict", response_model=PredictResponse)
def predict(request: PredictRequest = File(...)) -> PredictResponse:

    # This is the Where's Waldo image as an RGB matrix
    image = Image.open(io.BytesIO(request.file.read())).convert("RGB")

    loc, _ = walbrute.find_match(image, walbrute.TEMPLATES)

    # This is a dummy prediction - compute a real one
    prediction = {"x": loc[0], "y": loc[1]}

    return PredictResponse(**prediction)


@app.get("/api")
def hello():
    return {
        "uptime": get_uptime(),
        "service": settings.COMPOSE_PROJECT_NAME,
    }


@app.get("/")
def index():
    return HTMLResponse(
        render("static/index.html", host=settings.HOST_IP, port=settings.CONTAINER_PORT)
    )


if __name__ == "__main__":

    uvicorn.run("api:app", host=settings.HOST_IP, port=settings.CONTAINER_PORT)
