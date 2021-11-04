from loguru import logger
import uvicorn
from fastapi import FastAPI
from starlette.responses import HTMLResponse
import os
from collections import deque

import middleware.cors
import middleware.logging
from dtos.requests import PredictRequest
from dtos.responses import PredictResponse, ActionType
import tensorflow as tf

import numpy as np

import json
import time

from settings import Settings, DQNAgent, load_env
from static.render import render
from utilities.utilities import get_uptime
import random

from pydantic import BaseSettings

# --- Welcome to your Emily API! --- #
# See the README for guides on how to test it.

# Your API endpoints under http://yourdomain/api/...
# are accessible from any origin by default.
# Make sure to restrict access below to origins you
# trust before deploying your API to production.

load_env()
s = Settings()
app = FastAPI()
middleware.logging.setup(app, exclude_paths=['/api/predict'])
middleware.cors.setup(app)

@app.post('/api/predict', response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:

    if s.prev_dist == 320023023:
        prev_dist = request.distance

    # You receive the entire game state in the request object.
    # Read the game state and decide what to do in the next game tick.
    next_state = [request.velocity.x,
                    request.velocity.y,
                    request.sensors.back,
                    request.sensors.front,
                    # request.sensors.right_back,
                    # request.sensors.left_side,
                    # request.sensors.left_front,
                    # request.sensors.front,
                    # request.sensors.right_front,
                    # request.sensors.right_side
                    ]
    if next_state[2] is None:
        next_state[2] = 2
    else: next_state[2] = next_state[2] / 500
    if next_state[3] is None:
        next_state[3] = 2
    else: next_state[3] = next_state[3] / 500
    # for i in range(2, len(next_state)):
    #     next_state[i] = next_state[i] / 500
    next_state = tf.reshape(tf.convert_to_tensor(next_state, dtype=tf.float32), [1, len(next_state)])
    
    
    if s.agent.state is None:
        s.agent.state = next_state

    action = s.agent.act(
        s.agent.state
    )

    r1 = (request.distance - prev_dist) / 100
    r2 = -10 if request.did_crash else 0
    reward = r1 + r2
    
    s.agent.memorize(s.agent.state, action, reward, next_state, request.did_crash)
    s.agent.state = next_state


    prev_dist = request.distance

    if request.did_crash:
        # Episode is over
        state = np.reshape(s.agent.state, [1, s.state_size])
        s.episode += 1
        
        s.agent.update_target_model()
        print("Episode: {}/{}, distance: {}, e: {:.2}".format(s.episode, s.EPISODES, request.distance, s.agent.epsilon))
        if episode % 2 == 0:
            s.agent.save("./save/tesla.h5")
        
    if len(s.agent.memory) > s.batch_size:
        s.agent.replay(s.batch_size)

    return PredictResponse(
        action=s.agent.actions[action]
    )

@app.get('/api')
def hello():
    return {
        "uptime": get_uptime(),
        "service": s.COMPOSE_PROJECT_NAME,
    }


@app.get('/')
def index():
    return HTMLResponse(
        render(
            'static/index.html',
            host=s.HOST_IP,
            port=s.CONTAINER_PORT
        )
    )


if __name__ == '__main__':

    uvicorn.run(
        'api:app',
        host=s.HOST_IP,
        port=s.CONTAINER_PORT
    )
