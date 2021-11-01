"""
Generate training data for Waldo. 
Dimensions are initially 1500 x 1500 composed of 300x300 tiles with one tile having waldo in it 
"""
import pandas as pd
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path

TRAIN_DIR = Path("../../example-data/wheres-waldo")
IMG_DIR = TRAIN_DIR / "images"
ANN_DIR = IMG_DIR / "annotations"


def get_img_path(img_name: str) -> str:
    return str(IMG_DIR / img_name)


def show_img(img: np.ndarray):
    cv2.imshow("Show", img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def read_content(xml_file: str):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_boxes = []

    for boxes in root.iter("object"):

        filename = root.find("filename").text

        ymin, xmin, ymax, xmax = None, None, None, None

        ymin = int(boxes.find("bndbox/ymin").text)
        xmin = int(boxes.find("bndbox/xmin").text)
        ymax = int(boxes.find("bndbox/ymax").text)
        xmax = int(boxes.find("bndbox/xmax").text)

        list_with_single_boxes = [xmin, ymin, xmax, ymax]
        list_with_all_boxes.append(list_with_single_boxes)

    return filename, list_with_all_boxes


def create_waldo_crop(img_path: str, ann_dict, img_dims=(300, 300)) -> np.ndarray:
    img = cv2.imread(get_img_path(img_path))
    bbox = ann_dict[img_path]
    bbox_x_len = bbox[2] - bbox[0]
    new_x_min = bbox[0] - np.random.randint(0, img_dims[0] - bbox_x_len)
    new_x_max = new_x_min + img_dims[0]
    bboy_y_len = bbox[3] - bbox[1]
    new_y_min = bbox[1] - np.random.randint(0, img_dims[1] - bboy_y_len)
    new_y_max = new_y_min + img_dims[1]
    return img[new_y_min:new_y_max, new_x_min:new_x_max]


def draw_bbox(img_path, bboxes) -> np.ndarray:
    img = cv2.imread(get_img_path(img_path))
    bbox = bboxes[0]
    return cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)


def create_empty_crop(img_path: str, ann_dict, img_dims=(300, 300)) -> np.ndarray:
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
            return img[y_min:y_max, x_min:x_max]


annotations = list(ANN_DIR.glob("*.xml"))

ex_file, bboxes = read_content(annotations[0])

ann_list = [[None, None] for _ in annotations]
for i, annotation in enumerate(annotations):
    file_name, bboxes = read_content(annotation)
    ann_list[i][0] = file_name
    ann_list[i][1] = bboxes[0]

ann_dict = {item[0]: item[1] for item in ann_list}


# Creating random waldo crops
test_path = "1.jpg"
test_img = cv2.imread(get_img_path(test_path))
test_bbox = ann_dict[test_path]
img_dims = (300, 300)


crop_box = (new_x_min, new_y_min, new_x_max, new_y_max)

new_img = draw_bbox(test_path, [crop_box])


empty_crop = create_empty_crop("2.jpg", ann_dict)
ann_dict.keys()
show_img(empty_crop)
