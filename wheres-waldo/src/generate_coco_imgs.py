"""
Get images in coco-format for yolo
"""
import numpy as np
import cv2
from pathlib import Path

TRAIN_DIR = Path("../../example-data/wheres-waldo")
IMG_DIR = TRAIN_DIR / "images"
ANN_DIR = IMG_DIR / "annotations"


def get_img_path(img_name: str) -> str:
    return str(IMG_DIR / img_name)


def create_empty_crop(img_path: str, ann_dict, img_dims=(300, 300)) -> np.ndarray:
    """ Creates a crop without waldo in it """
    img = cv2.imread(get_img_path(img_path))
    bbox = ann_dict[img_path]
    while True:
        x_min = np.random.randint(0, img.shape[1] - img_dims[0])
        y_min = np.random.randint(0, img.shape[0] - img_dims[1])
        x_max = x_min + img_dims[0]
        y_max = y_min + img_dims[1]
        # Check that there's no waldo
        to_the_left = x_min > bbox[2]
        above = y_min > bbox[3]
        to_the_right = x_max < bbox[0]
        below = y_max < bbox[1]
        if to_the_left or above or to_the_right or below:
            new_img = img[y_min:y_max, x_min:x_max]
            if new_img.shape == (300, 300, 3):
                return new_img


def create_waldo_crop(img_path: str, ann_dict, img_dims=(300, 300)) -> np.ndarray:
    """ Creates a picture with img_dims with waldo in it"""
    img = cv2.imread(get_img_path(img_path))
    bbox = ann_dict[img_path]
    bbox_x_len = bbox[2] - bbox[0]
    retry_count = 0
    while True:
        rand_x = np.random.randint(0, img_dims[0] - bbox_x_len)
        new_x_min = bbox[0] - rand_x
        new_x_max = new_x_min + img_dims[0]
        bbox_y_len = bbox[3] - bbox[1]
        rand_y = np.random.randint(0, img_dims[1] - bbox_y_len)
        new_y_min = bbox[1] - rand_y
        new_y_max = new_y_min + img_dims[1]
        new_bbox = [rand_x, rand_y, rand_x + bbox_x_len, rand_y + bbox_y_len]
        new_img = img[new_y_min:new_y_max, new_x_min:new_x_max]
        if new_img.shape == (img_dims[0], img_dims[1], 3):
            return new_img, new_bbox
        print(f"problems with {img_path}")
        retry_count += 1
