"""
Formats the iq-images into a list of list of images, with sublists being rows and images being figures
"""
import cv2
import numpy as np
import itertools
from pathlib import Path
from typing import List

IMG_DIR = Path("../../example-data/iq-test/dmi-api-test/")


def read_img(path: Path):
    return cv2.imread(str(path))


def remove_arrows(img_list: List[List[np.ndarray]]) -> List[List[np.ndarray]]:
    return [[a for i, a in enumerate(seq) if i % 2 == 0] for seq in img_list]


def split_img(img: np.ndarray, fig_size=(110, 110)) -> List[List[np.ndarray]]:
    x_len = img.shape[1]
    y_len = img.shape[0]
    h = int(y_len / fig_size[1])
    w = int(x_len / fig_size[0])
    positions = itertools.product(range(h), range(w))
    result_list = [[None for _ in range(w)] for _ in range(h)]
    i = 0
    for h_pos, w_pos in positions:
        x = w_pos * fig_size[0]
        y = h_pos * fig_size[1]
        new_img = img[y : y + fig_size[1], x : x + fig_size[0], :]
        try:
            result_list[h_pos][w_pos] = new_img
        except IndexError:
            print(f"{i=}")
    final_list = remove_arrows(result_list)
    return final_list


def show_img(img):
    cv2.imshow("show", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


task_img_paths = list(IMG_DIR.glob("*image*.png"))

img = read_img(task_img_paths[0])

show_img(img)

img_list = split_img(img)

show_img(img_list[0][2])

