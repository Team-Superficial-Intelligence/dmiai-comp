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
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import json
import time

from settings import Settings, load_env, ModelState
from static.render import render
from utilities.utilities import get_uptime
import random

from pydantic import BaseSettings

load_env()
settings = Settings()
settings.state_size = 5
settings.action_size = 5
settings.model_path = './model.h5'
settings.gamma = 0.95
settings.learning_rate = 0.001


def create_dqn_model():
    model = keras.Sequential([
        layers.Dense(settings.state_size, activation='relu',
                     input_shape=(1, settings.state_size)),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(settings.action_size, activation='linear')
    ])
    model.compile(loss='mse', optimizer=keras.optimizers.Adam(
        lr=settings.learning_rate))
    return model


def save_model(model):
    model.save_weights(settings.model_path)


def initialise_model():
    model = create_dqn_model()
    # Check if file exists
    if os.path.isfile(settings.model_path):
        model.load_weights(settings.model_path)
    return model


def dqn():
    """
    Create a deep Q-learning model and train it on the given data.
    """
    # Initialise model
    model = initialise_model()

    # Initialise memory
    memory = []

    # Initialise variables
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    done = False
    step = 0
    state = None
    score = 0
    last_score = 0
    last_action = None
    last_state = None
    last_reward = None

    while not done:
        # Get state
        state = get_state(model)

        # Choose action
        action = choose_action(state, epsilon)

        # Get reward
        reward = get_reward(action)

        # Store transition
        store_transition(memory, state, action, reward)

        # Train the model
        if len(memory) > settings.batch_size:
            learn(model, memory)

        # Update variables
        last_score = score
        last_action = action
        last_state = state
        last_reward = reward
        score += reward
        step += 1
        epsilon = max(epsilon_min, epsilon_decay * epsilon)

        # Check if done
        done = check_done(step, score)

    # Save model
    save_model(model)

    return model


def store_transition(memory, state, action, reward):
    memory.append((state, action, reward))


def learn(model, memory):
    """
    Train the model on the given data.
    """
    # Initialise variables
    batch = random.sample(memory, settings.batch_size)
    states = np.array([transition[0] for transition in batch])
    actions = np.array([transition[1] for transition in batch])
    rewards = np.array([transition[2] for transition in batch])

    # Calculate target
    target = rewards + settings.gamma * np.amax(model.predict(
        np.array([next_state])), axis=1)

    # Train the model
    model.fit(states, target, epochs=1, verbose=0)


def get_state(model):
    """
    Get the state of the game.
    """
    state = np.array([[random.randint(0, settings.state_size - 1)]])
    return state


def choose_action(state, epsilon):
    """
    Choose an action.
    """
    if random.random() <= epsilon:
        action = random.randint(0, settings.action_size - 1)
    else:
        action = np.argmax(model.predict(state))
    return action


def get_reward(action):
    """
    Get the reward for the given action.
    """
    if action == 0:
        reward = 1
    else:
        reward = 0
    return reward


# --- Welcome to your Emily API! --- #
# See the README for guides on how to test it.

# Your API endpoints under http://yourdomain/api/...
# are accessible from any origin by default.
# Make sure to restrict access below to origins you
# trust before deploying your API to production.


app = FastAPI()

middleware.logging.setup(app, exclude_paths=['/api/predict'])
middleware.cors.setup(app)

# Set up the model
model = initialise_model()
model_target = initialise_model()
model_state = ModelState()
state_next = []


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

    state = [x / 1000 if x is not None else 1 for x in state]
    state = np.reshape(state, [1, 5])
    state = tf.convert_to_tensor(state, dtype=tf.float32)

    # logger.info(state)

    #### TEMP LOGGING ####
    req_group = str(time.time_ns())
    f = open("/workspace/rq_1.json", "a")
    json.dump(request, f, default=repr)
    f.close()
    #### /TEMP LOGGING ####

    # Get something along those lines of:
    # state_next, reward, done, _ = env.step(action)
    # state_next = np.array(state_next)

    model_state.step_count += 1

    # Use epsilon-greedy to select an action
    if random.random() < model_state.epsilon:
        action = random.choice(model_state.actions)
    else:
        state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
        action_probs = model(state_tensor, training=False)
        # Take best action
        action = tf.argmax(action_probs[0]).numpy()
        action = model_state.actions[action]

    # Decay probability of taking a random action
    model_state.epsilon = max(model_state.epsilon_min,
                              model_state.epsilon * model_state.epsilon_decay)

    reward = 0
    reward += request.velocity.x

    # Give giant punishment for crashing lmao
    if request.did_crash:
        reward -= 10000
        logger.info(f'Crashed after {request.elapsed_time_ms} ms')

    # Save actions and states in replay buffer
    # find index of action in actions
    action_index = model_state.actions.index(action)
    model_state.action_history.append(action_index)
    if len(model_state.next_state_history) > 0:
        model_state.state_history.append(model_state.next_state_history[-1])
    else:
        model_state.state_history.append(state)
    model_state.next_state_history.append(state)
    model_state.done_history.append(request.did_crash)
    model_state.reward_history.append(reward)

    # Only update the model if we have a large enough batch_size
    if model_state.step_count % model_state.update_after_actions == 0 and len(model_state.done_history) > model_state.batch_size:
        # Get indices of samples for replay buffers
        indices = np.random.choice(
            range(len(model_state.done_history)), size=model_state.batch_size)

        # Using list comprehension to sample from replay buffer
        state_sample = np.array(
            [model_state.state_history[i] for i in indices])
        state_next_sample = np.array(
            [model_state.next_state_history[i] for i in indices])
        rewards_sample = [model_state.reward_history[i] for i in indices]
        action_sample = [model_state.action_history[i] for i in indices]
        done_sample = tf.convert_to_tensor(
            [float(model_state.done_history[i]) for i in indices]
        )

        # Build the updated Q-values for the sampled future states
        # Use the target model for stability
        future_rewards = model_target.predict(state_next_sample)

        # Get largest value for each row in the future rewards
        future_rewards = np.max(future_rewards, axis=2)

        # Q value = reward + discount factor * expected future reward
        # logger.info(f'Future rewards: {future_rewards}')
        updated_q_values = rewards_sample + settings.gamma * tf.reduce_max(
            future_rewards, axis=1
        )

        # logger.info(f'Updated Q values: {updated_q_values}')

        # Create a mask so we only calculate loss on the updated Q-values
        masks = tf.one_hot(action_sample, settings.action_size)

        with tf.GradientTape() as tape:
            # Train the model on the states and updated Q-values
            q_values = model(state_sample)
            q_values = np.max(q_values, axis=2)

            # Apply the masks to the Q-values to get the Q-value for action taken
            q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
            # logger.info(f'Q-values: {q_values}')
            # Calculate loss between new Q-value and old Q-value
            # loss = tf.reduce_mean(tf.square(updated_q_values - q_action))
            loss = model_state.loss_function(updated_q_values, q_action)

        # Backpropagation
        logger.info(f'Loss: {model.trainable_variables}')
        grads = tape.gradient(loss, model.trainable_variables)
        # logger.warning(
        #     f'Grads: {grads}\nTrainable parameters: {model.trainable_variables}')
        model_state.optimizer.apply_gradients(
            zip(grads, model.trainable_variables))

    # Update model weights if we got far enough
    if model_state.step_count % model_state.update_target_network == 0:
        # update the the target network with new weights
        model_target.set_weights(model.get_weights())
        # Log details
        template = "running reward: {:.2f} at episode {}, frame count {}"
        print(template.format(model_state.running_reward,
                              model_state.episode_count,
                              model_state.step_count))

    # Limit the state and reward history
    if len(model_state.reward_history) > model_state.max_memory_length:
        del model_state.reward_history[:1]
        del model_state.state_history[:1]
        del model_state.state_next_history[:1]
        del model_state.action_history[:1]
        del model_state.done_history[:1]

    # Update running reward to check how well we're doing
    model_state.episode_reward_history.append(model_state.running_reward)
    if len(model_state.episode_reward_history) > 100:
        del model_state.episode_reward_history[:1]
    model_state.running_reward = np.mean(model_state.episode_reward_history)

    # supress errors
    try:
        return PredictResponse(
            action=action
        )
    except:
        logger.warning("Failed to return a proper response.")
        return PredictResponse(
            action=ActionType.NOTHING
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
