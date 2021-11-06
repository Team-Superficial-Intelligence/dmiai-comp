from pydantic import BaseSettings
import random
from utilities.utilities import get_uptime
from static.render import render
from settings import Settings, load_env
import time
import json
import numpy as np
import middleware.logging
import middleware.cors
from loguru import logger
import uvicorn
from fastapi import FastAPI
from starlette.responses import HTMLResponse
import os
import bcolors as b

from dtos.responses import PredictResponse, ActionType
from dtos.requests import PredictRequest


load_env()
settings = Settings()
app = FastAPI()

middleware.logging.setup(app, exclude_paths=['/api/predict'])
middleware.cors.setup(app)

# Actions = avoid car front
settings.start_timing = 0
settings.avoid_front = False
settings.block_avoid_front_change = False
settings.action_timing = 0
settings.action_choice = ActionType.ACCELERATE


@app.post('/api/predict', response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:

    # You receive the entire game state in the request object.
    # Read the game state and decide what to do in the next game tick.

    # print("Getting loop.")
    # print(str(PredictRequest["sensors"]["left_side"]))

    # Implicitly bias towards accelerating because of reward
    # Set the state = sensors on the front of the car

    actions = [ActionType.ACCELERATE, ActionType.NOTHING,
               ActionType.STEER_LEFT, ActionType.STEER_RIGHT,
               ActionType.DECELERATE]

    if request.sensors.front is None: request.sensors.front = 1000 
    if request.sensors.back is None: request.sensors.back = 1000 

    action = ActionType.ACCELERATE

    if settings.start_timing < 0:
        settings.avoid_front = False    
        
    # Stay in lane after car
    if request.sensors.front > 950:
        action = ActionType.ACCELERATE
    elif request.sensors.front > 800:
        action = ActionType.NOTHING
    else:
        action = ActionType.DECELERATE
    
    if action == ActionType.ACCELERATE and request.velocity.x > 160:
        action = ActionType.NOTHING

    print(settings.step)
    if request.sensors.front < 600 and request.velocity.x < 40:
        settings.start_timing = settings.step
        settings.avoid_front = True
    elif request.sensors.back < 200 and request.sensors.front >950:
        action = ActionType.ACCELERATE
    elif request.sensors.back < 200:
        settings.start_tming = settings.step
        settings.avoid_front = True

    if settings.avoid_front:
        if settings.step in range(settings.start_timing, settings.start_timing+5):
            action = ActionType.STEER_LEFT
        elif settings.step in range(settings.start_timing+5, settings.start_timing+11):
            action = ActionType.STEER_RIGHT
        elif settings.step < settings.start_timing+31:
            action = ActionType.NOTHING
        else:
            settings.avoid_front = False
            if settings.lane == 1:
                settings.lane = 0
        print(f'{b.OK}Action step: {settings.step}, action: {action}{b.END}')

    settings.step += 1    
    if request.did_crash or request.elapsed_time_ms < 10:
        settings.results_log.append(
            {
                'step': settings.step,
                'action': action,
                'crash': request.did_crash,
                'time': request.elapsed_time_ms
            }
        )
        print(f'Results {settings.results_log[-1]}')
        settings.step = 0
    

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
