
from loguru import logger
import uvicorn
from fastapi import FastAPI
from starlette.responses import HTMLResponse

import middleware.cors
import middleware.logging
import middleware.contenttype
from dtos.requests import PredictRequest
from dtos.responses import PredictResponse, ActionType

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
# force json request content type
app.add_middleware(middleware.contenttype.ForceJSONContentTypeMiddleware)
middleware.logging.setup(app, exclude_paths=['/api/predict'])
middleware.cors.setup(app)



@app.post('/api/predict', response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:

    if request.velocity.x < 80:
        raction = ActionType.ACCELERATE
    else:
        raction = ActionType.DECELERATE

    if request.did_crash:
        logger.info(f'Crashed after {request.elapsed_time_ms} ms')
    return PredictResponse(action=raction)
    # You receive the entire game state in the request object.
    # Read the game state and decide what to do in the next game tick.

    if request.did_crash:
        logger.info(f'Crashed after {request.elapsed_time_ms} ms')

    actions = [ActionType.ACCELERATE, ActionType.DECELERATE,
               ActionType.STEER_LEFT, ActionType.STEER_RIGHT,
               ActionType.NOTHING]

    return PredictResponse(
        action=random.choice(actions)
    )


@app.get('/api')
def hello():
    return {
        "uptime": get_uptime(),
        "service": settings.COMPOSE_PROJECT_NAME,
    }


@app.get('/')
def index():
    return HTMLResponse(
        render(
            'static/index.html',
            host=settings.HOST_IP,
            port=settings.CONTAINER_PORT
        )
    )


if __name__ == '__main__':

    uvicorn.run(
        'api:app',
        host=settings.HOST_IP,
        port=settings.CONTAINER_PORT,
        debug=True
    )
