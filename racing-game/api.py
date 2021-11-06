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
<<<<<<< HEAD

middleware.logging.setup(app, exclude_paths=['/api/predict'])
middleware.cors.setup(app)

# Actions = avoid car front
settings.start_timing = 0
settings.avoid_front = False
settings.block_avoid_front_change = False
settings.action_timing = 0
settings.action_choice = ActionType.ACCELERATE


@app.post('/api/predict', response_model=PredictResponse)
=======
middleware.logging.setup(app, exclude_paths=["/api/predict"])
middleware.cors.setup(app)


@app.post("/api/predict", response_model=PredictResponse)
>>>>>>> f977b90c847e343e84b77916015e21416f72a38d
def predict(request: PredictRequest) -> PredictResponse:

    # You receive the entire game state in the request object.
    # Read the game state and decide what to do in the next game tick.
<<<<<<< HEAD

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
=======
    next_state = [
        request.velocity.x,
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
    else:
        next_state[2] = next_state[2] / 500
    if next_state[3] is None:
        next_state[3] = 2
    else:
        next_state[3] = next_state[3] / 500
    # for i in range(2, len(next_state)):
    #     next_state[i] = next_state[i] / 500
    next_state = tf.reshape(
        tf.convert_to_tensor(next_state, dtype=tf.float32), [1, len(next_state)]
    )

    if s.agent.state is None:
        s.agent.state = next_state

    action = s.agent.act(s.agent.state)

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
        print(
            "Episode: {}/{}, distance: {}, e: {:.2}".format(
                s.episode, s.EPISODES, request.distance, s.agent.epsilon
            )
        )
        if episode % 2 == 0:
            s.agent.save("./save/tesla.h5")

    if len(s.agent.memory) > s.batch_size:
        s.agent.replay(s.batch_size)

    return PredictResponse(action=s.agent.actions[action])


@app.get("/api")
>>>>>>> f977b90c847e343e84b77916015e21416f72a38d
def hello():
    return {
        "uptime": get_uptime(),
        "service": settings.COMPOSE_PROJECT_NAME,
    }


@app.get("/")
def index():
    return HTMLResponse(
<<<<<<< HEAD
        render(
            'static/index.html',
            host=settings.HOST_IP,
            port=settings.CONTAINER_PORT
        )
=======
        render("static/index.html", host=s.HOST_IP, port=s.CONTAINER_PORT)
>>>>>>> f977b90c847e343e84b77916015e21416f72a38d
    )


if __name__ == "__main__":

<<<<<<< HEAD
    uvicorn.run(
        'api:app',
        host=settings.HOST_IP,
        port=settings.CONTAINER_PORT
    )
=======
    uvicorn.run("api:app", host=s.HOST_IP, port=s.CONTAINER_PORT)
>>>>>>> f977b90c847e343e84b77916015e21416f72a38d
