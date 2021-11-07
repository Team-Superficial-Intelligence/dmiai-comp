from typing import Optional
from pydantic import BaseSettings
from dotenv import load_dotenv
from argparse import ArgumentParser
from dtos.responses import ActionType


class Settings(BaseSettings):

    IPC: str
    RUNTIME: str
    HOST_IP: str
    HOST_PORT: int
    CONTAINER_PORT: int
    COMPOSE_PROJECT_NAME: str
    NVIDIA_VISIBLE_DEVICES: Optional[str]
    NVIDIA_DRIVER_CAPABILITIES: Optional[str]
    start_timing: Optional[int]
    action_timing: Optional[int]
    avoid_front: Optional[bool]
    block_avoid_front_change: Optional[bool]
    action_choice = ActionType.ACCELERATE
    step = 0
    results_log = []
    lane = 0
    auto = "left"
    last_distance = 0
    total_distance = 0
    episode = 0


def load_env():
    # Let the script caller define the .env file to use, e.g.:  python api.py --env .prod.env
    parser = ArgumentParser()
    parser.add_argument('-e', '--env', default='.env',
                        help='Sets the environment file')

    args = parser.parse_args()
    load_dotenv(args.env)
