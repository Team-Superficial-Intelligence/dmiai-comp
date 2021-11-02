"""
Get images in coco-format for yolo
"""
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List

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


def read_content(xml_file: str):
    """ Parses a XML file of PASCAL VOC Annotations"""
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
    return filename, list_with_all_boxes[0]


def create_ann_dict(annotations: List[Path]) -> Dict[str, List[int]]:
    ann_list = [[None, None] for _ in annotations]
    for i, annotation in enumerate(annotations):
        file_name, bboxes = read_content(annotation)
        ann_list[i][0] = file_name
        ann_list[i][1] = bboxes
    return {item[0]: item[1] for item in ann_list}


annotations = list(ANN_DIR.glob("*.xml"))
ann_dict = create_ann_dict(annotations)
test_img = cv2.imread(get_img_path("1.jpg"))
