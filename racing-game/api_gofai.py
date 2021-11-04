from loguru import logger
import uvicorn
from fastapi import FastAPI
from starlette.responses import HTMLResponse
import os

import middleware.cors
import middleware.logging
from dtos.requests import PredictRequest
from dtos.responses import PredictResponse, ActionType

import numpy as np

import json
import time

from settings import Settings, load_env
from static.render import render
from utilities.utilities import get_uptime
import random

from pydantic import BaseSettings

load_env()
settings = Settings()
app = FastAPI()

middleware.logging.setup(app, exclude_paths=['/api/predict'])
middleware.cors.setup(app)

# Actions = avoid car front
settings.start_timing = 0
settings.avoid_front = False
settings.action_timing = 0


@app.post('/api/predict', response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:

    # You receive the entire game state in the request object.
    # Read the game state and decide what to do in the next game tick.

    # print("Getting loop.")
    # print(str(PredictRequest["sensors"]["left_side"]))

    # Implicitly bias towards accelerating because of reward
    # Set the state = sensors on the front of the car

    state = np.array([request.sensors.left_side,
                      request.sensors.left_front,
                      request.sensors.front,
                      request.sensors.right_front,
                      request.sensors.right_side])

    if state[2] is None:
        state[2] = 1000

    print("State: " + str(state))

    allowable_actions = [ActionType.ACCELERATE, ActionType.NOTHING,
                         ActionType.STEER_LEFT, ActionType.STEER_RIGHT,
                         ActionType.DECELERATE]

    action = ActionType.ACCELERATE

    # Select action GOFAI
    if not settings.avoid_front:
        if state[2] < 1000:
            action = ActionType.DECELERATE
            settings.avoid_front = True
            settings.start_timing = request.elapsed_time_ms
        if not settings.avoid_front and request.velocity.x < 75:
            action = ActionType.ACCELERATE

    if settings.avoid_front:
        print(f'Avoiding front, current time {settings.action_timing}')
        settings.action_timing += request.elapsed_time_ms - settings.start_timing
        # Decelerate for 1 second
        if settings.action_timing < 1000:
            action = ActionType.DECELERATE

        # Steer towards clear area
        if settings.action_timing > 1000 and settings.action_timing < 2000:
            if state[1] < 600 and state[0] > 400:
                action = ActionType.STEER_RIGHT
            elif state[4] > 400 and state[3] < 600:
                action = ActionType.STEER_LEFT

        # 2 seconds to avoid front
        if settings.action_timing > 2000:
            settings.avoid_front = False
            settings.action_timing = 0

    return PredictResponse(
        action=action
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
        port=settings.CONTAINER_PORT
    )
