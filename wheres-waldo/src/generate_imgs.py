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


def draw_bbox(img_path, bboxes) -> np.ndarray:
    img = cv2.imread(get_img_path(img_path))
    bbox = bboxes[0]
    return cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)


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


def create_waldo_crop(img_path: str, img_dims=(300, 300)) -> np.ndarray:

    img = cv2.imread(get_img_path(img_path))
    bbox_x_len = test_bbox[2] - test_bbox[0]
    new_x_min = test_bbox[0] - np.random.randint(0, img_dims[0] - bbox_x_len)
    new_x_max = new_x_min + img_dims[0]
    bboy_y_len = test_bbox[3] - test_bbox[1]
    new_y_min = test_bbox[1] - np.random.randint(0, img_dims[1] - bboy_y_len)
    new_y_max = new_y_min + img_dims[1]

    return img[new_y_min:new_y_max, new_x_min:new_x_max]

