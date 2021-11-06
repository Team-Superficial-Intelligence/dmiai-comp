from typing import Optional, Deque
from pydantic import BaseSettings
from dotenv import load_dotenv
from argparse import ArgumentParser
from tensorflow import keras
from tensorflow.keras import losses
from dtos.responses import ActionType

from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import numpy as np

import random
import tensorflow as tf

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        self.actions = [ActionType.ACCELERATE, ActionType.DECELERATE, ActionType.STEER_LEFT, ActionType.STEER_RIGHT, ActionType.NOTHING]
        self.state = None

    """Huber loss for Q Learning

    References: https://en.wikipedia.org/wiki/Huber_loss
                https://www.tensorflow.org/api_docs/python/tf/losses/huber_loss
    """

    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * \
            K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        inputs = Input(shape=(self.state_size,))
        layer1 = Dense(24, activation='relu')(inputs)
        action = Dense(self.action_size, activation='linear')(layer1)

        return keras.Model(inputs=inputs, outputs=action)
        model.compile(loss=self._huber_loss,
                      optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            print(state)
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                # a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
                # target[0][action] = reward + self.gamma * t[np.argmax(a)]
            self.model.fit(state, target, epochs=1, verbose=1)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

class Settings(BaseSettings):

    IPC: str
    RUNTIME: str
    HOST_IP: str
    HOST_PORT: int
    CONTAINER_PORT: int
    COMPOSE_PROJECT_NAME: str
    NVIDIA_VISIBLE_DEVICES: Optional[str]
    NVIDIA_DRIVER_CAPABILITIES: Optional[str]
    model_path: Optional[str]
    model_name: Optional[str]
    state_size: Optional[int]
    action_size: Optional[int]
    gamma: Optional[float]
    learning_rate: Optional[float]
    batch_size: Optional[int]
    memory_size: Optional[int]
    avoid_front: Optional[bool]
    action_timing: Optional[float]
    start_timing: Optional[float]
    EPISODES = 100
    state_size = 4
    action_size = 2
    agent = DQNAgent(state_size, action_size)
    episode = 0
    batch_size = 32
    prev_dist = 320023023


def load_env():
    # Let the script caller define the .env file to use, e.g.:  python api.py --env .prod.env
    parser = ArgumentParser()
    parser.add_argument('-e', '--env', default='.env',
                        help='Sets the environment file')

    args = parser.parse_args()
    load_dotenv(args.env)
