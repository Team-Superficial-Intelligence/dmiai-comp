import numpy as np
import shit_checker.matrix_solver as ms
import pprint
from pathlib import Path
import shit_checker.format_iq_imgs as fii
import shit_checker.rotate_check as rc

Blank = 0
Red = 10
Blue = 100
Green = 1000
Yellow = 10000

# Combinations
Red_combi = [10, 20, 30, 40, 120, 130, 1020, 1030, 1120, 10020, 10030, 11020, 10120]
Blue_combi = [
    100,
    200,
    300,
    400,
    210,
    310,
    1200,
    1300,
    1210,
    10200,
    10300,
    11200,
    10210,
]
Green_combi = [
    1000,
    2000,
    3000,
    4000,
    2010,
    3010,
    2110,
    2100,
    3100,
    12000,
    13000,
    12100,
    12010,
]
Yellow_combi = [
    10000,
    20000,
    30000,
    40000,
    20010,
    30010,
    20100,
    30100,
    21000,
    31000,
    20110,
    21010,
    21100,
]
Blank_combi = [
    0,
    110,
    220,
    1010,
    2020,
    1100,
    2200,
    10010,
    20020,
    10100,
    20200,
    11000,
    22000,
]


def next_step(world: np.ndarray) -> np.ndarray:
    neighbor = np.zeros(world.shape, dtype=int)
    neighbor[1:] += world[:-1]  # North
    neighbor[:-1] += world[1:]  # South
    neighbor[:, 1:] += world[:, :-1]  # West
    neighbor[:, :-1] += world[:, 1:]  # East
    # Replacing the colors
    neighbor[np.isin(neighbor, Red_combi)] = Red
    neighbor[np.isin(neighbor, Blue_combi)] = Blue
    neighbor[np.isin(neighbor, Green_combi)] = Green
    neighbor[np.isin(neighbor, Yellow_combi)] = Yellow
    neighbor[np.isin(neighbor, Blank_combi)] = Blank
    return neighbor


if __name__ == "__main__":
    # reading the image in grayscale mode
    IMG_DIR = Path("../example-data/iq-test/dmi-api-test")
    IMG_DIR.exists()
    img_files = list(IMG_DIR.glob("*image*.png"))
    img_path = img_files[4]
    test_img = fii.read_img(img_path)
    image_list = fii.split_img(test_img)
    choices = [
        fii.read_img(choice)
        for choice in rc.find_img_choices(img_path, img_dir=IMG_DIR)
    ]
    matrix_list = ms.check_matrix(image_list, choices)
    test_shit
    # Image to alter

pprint.pprint(world)
pprint.pprint(neighbor)

