from typing import List, Tuple
from pathlib import Path
import cv2
import torch
import itertools
import numpy as np

TEST_PATH = Path("../example-data/wheres-waldo/waldo.jpg")
MODEL = torch.hub.load("ultralytics/yolov5", "custom", path="waldominator_v1.pt")


def show_img(img):
    cv2.imshow("shown", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def PIL_to_cv2(pil_image):
    """ Converts PIL image to cv2 format (BGR) """
    cv2_img = np.array(pil_image)
    return cv2_img[:, :, ::-1].copy()


def crop_img(img: np.ndarray, pos: Tuple[int], img_size=(300, 300)) -> np.ndarray:
    x = pos[1] * img_size[1]
    y = pos[0] * img_size[0]
    return img[y : y + img_size[0], x : x + img_size[1]]


def split_img_matrix(img: np.ndarray, img_size=(300, 300)) -> List[np.ndarray]:
    width = int(img.shape[1] / img_size[1])
    height = int(img.shape[0] / img_size[0])
    positions = list(itertools.product(range(height), range(width)))
    return [crop_img(img, pos) for pos in positions], positions


test_img = cv2.imread(str(TEST_PATH))

img_list, positions = split_img_matrix(test_img)

model_output = MODEL(img_list)

model_output.pred
