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

from dtos.responses import PredictResponse
from dtos.responses import ActionType as a
from dtos.requests import PredictRequest


def rename_attribute(object_, old_attribute_name, new_attribute_name):
    setattr(object_, new_attribute_name, getattr(object_, old_attribute_name))
    delattr(object_, old_attribute_name)


load_env()
s = Settings()
app = FastAPI()

middleware.logging.setup(app, exclude_paths=['/api/predict'])
middleware.cors.setup(app)

# Actions = avoid car front
s.start_timing = 0
s.avoid_front = False
s.action_timing = 0
s.action_choice = a.ACCELERATE
s.lane = "mid"
s.total_distance = 0
s.last_distance = 0


@app.post('/api/predict', response_model=PredictResponse)
def predict(response: PredictRequest) -> PredictResponse:

    r = response

    # You receive the entire game state in the request object.
    # Read the game state and decide what to do in the next game tick.

    # print("Getting loop.")
    # print(str(PredictRequest["sensors"]["left_side"]))

    # Implicitly bias towards accelerating because of reward
    # Set the state = sensors on the front of the car

    actions = [a.ACCELERATE, a.NOTHING,
               a.STEER_LEFT, a.STEER_RIGHT,
               a.DECELERATE]

    if r.sensors.front is None:
        r.sensors.front = 1000
    if r.sensors.back is None:
        r.sensors.back = 1000
    if r.sensors.left_front is None:
        r.sensors.left_front = 1000
    if r.sensors.right_front is None:
        r.sensors.right_front = 1000
    if r.sensors.left_back is None:
        r.sensors.left_back = 1000
    if r.sensors.right_back is None:
        r.sensors.right_back = 1000

    action = a.ACCELERATE
    front_lim = 900
    speed_lim = 1000

    if s.start_timing < 0:
        s.avoid_front = False

    # print(s.step)
    if not s.avoid_front:
        # Stay in lane after car
        if r.sensors.front > front_lim:
            action = a.ACCELERATE
        else:
            action = a.DECELERATE

        # Handle shifting lanes action start
        if r.sensors.front < front_lim and r.velocity.x < speed_lim or r.sensors.back < 200:
            # Check sensors if there is space on the left
            print(f'Eval: {s.lane}')
            if s.lane == "mid":
                if r.sensors.left_side < 300 or r.sensors.left_front < 500:
                    s.auto = "right"
                    s.lane = "right"
                else:
                    s.auto = "left"
                    s.lane = "left"
            elif s.lane == "left":
                if r.sensors.right_side < 300 or r.sensors.right_front < 500:
                    s.auto = "decelerate"
                else:
                    s.auto = "right"
                    s.lane = "mid"
            elif s.lane == "right":
                s.auto = "left"
                s.lane = "mid"
            s.start_timing = s.step
            s.avoid_front = True
            print(
                f'{b.OK}Step: {s.step - s.start_timing}, lane: {s.lane}, action: {s.auto}{b.END}')
        elif r.sensors.back < 200 and r.sensors.front > front_lim:
            action = a.ACCELERATE
    else:
        # Do that action yo
        if s.auto == "left" or s.auto == "right":
            if s.step < s.start_timing + 15:
                action = a.STEER_LEFT if s.auto == "left" else a.STEER_RIGHT
            elif s.step <= s.start_timing + 20:
                if r.sensors.front > front_lim:
                    action = a.ACCELERATE
                else:
                    action = a.DECELERATE
            elif s.step > s.start_timing + 20 and r.velocity.y != 0:
                if s.auto == "left":
                    if r.velocity.y < 0:
                        action = a.STEER_RIGHT
                    elif r.velocity.y > 0:
                        ation = a.STEER_LEFT
                elif s.auto == "right":
                    if r.velocity.y > 0:
                        action = a.STEER_LEFT
                    elif r.velocity.y < 0:
                        ation = a.STEER_RIGHT
            elif s.step > s.start_timing + 40:
                s.avoid_front = False
        elif s.auto == "decelerate":
            action = a.DECELERATE
            s.avoid_front = False

    if s.step % 500 == 0 and s.step != 0:
        print(
            f'{b.OKMSG}Report:\n\tStep: {s.step}\n\tDistance: {r.distance}\n\tElapsed time: {r.elapsed_time_ms}\n\tSpeed: {r.velocity.x}\n\tTotal distance: {s.total_distance}{b.END}\n')
        s.total_distance += r.distance - s.last_distance
        s.last_distance = r.distance

    s.step += 1
    if r.did_crash or r.elapsed_time_ms < 10:
        s.results_log.append(
            {
                'steps': s.step,
                'last_action': action,
                'did_crash': r.did_crash,
                'distance': r.distance,
                'total_distance': s.total_distance
            }
        )
        print(f'Results {s.results_log[-1]}')
        s.episode += 1
        s.step = 0
        s.avoid_front = False

    return PredictResponse(
        action=action
    )


@ app.get('/api')
def hello():
    return {
        "uptime": get_uptime(),
        "service": s.COMPOSE_PROJECT_NAME,
    }


@ app.get("/")
def index():
    return HTMLResponse(
        render(
            'static/index.html',
            host=s.HOST_IP,
            port=s.CONTAINER_PORT
        )
    )


if __name__ == "__main__":

    uvicorn.run(
        'api:app',
        host=s.HOST_IP,
        port=s.CONTAINER_PORT
    )
