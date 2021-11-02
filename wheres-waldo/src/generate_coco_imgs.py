"""
Get images in coco-format for yolo
"""
import numpy as np
import cv2
import random
import string
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Union

TRAIN_DIR = Path("../../example-data/wheres-waldo")
IMG_DIR = TRAIN_DIR / "images"
ANN_DIR = IMG_DIR / "annotations"
DATA_DIR = Path.home() / "datasets" / "waldo"
NEW_IMG_DIR = DATA_DIR / "images"
TRAIN_IMG_DIR = NEW_IMG_DIR / "train"
VAL_IMG_DIR = NEW_IMG_DIR / "val"
LABEL_DIR = DATA_DIR / "labels"
TRAIN_LABEL_DIR = LABEL_DIR / "train"
VAL_LABEL_DIR = LABEL_DIR / "val"


def get_img_path(img_name: str) -> str:
    return str(IMG_DIR / img_name)


def create_random_id(N=8):
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=N))


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


def format_yolo_bbox(bbox: List[int], img_dims=(300, 300)) -> List[float]:
    x_mid = ((bbox[0] + bbox[2]) / 2) / img_dims[0]
    y_mid = ((bbox[1] + bbox[3]) / 2) / img_dims[1]
    width = (bbox[2] - bbox[0]) / img_dims[0]
    height = (bbox[3] - bbox[1]) / img_dims[1]
    return [x_mid, y_mid, width, height]


def format_yolo_string(bbox: List[float]) -> str:
    full_dat = ["0"] + bbox
    return "\t".join(str(x) for x in full_dat)


def write_text(s: str, file_path: Path):
    with open(file_path, "w") as f:
        f.write(s)


annotations = list(ANN_DIR.glob("*.xml"))
ann_dict = create_ann_dict(annotations)
test_img = cv2.imread(get_img_path("1.jpg"))

val_idx = int(len(ann_dict) * 0.8)
for i, (img_name, bbox) in enumerate(ann_dict.items()):
    if img_name == "21.jpg":
        continue
    for j in range(3):
        waldo_img, waldo_bbox = create_waldo_crop(img_name, ann_dict)
        yolo_bbox = format_yolo_bbox(waldo_bbox)
        file_id = create_random_id()
        if i <= val_idx:
            write_text(
                format_yolo_string(yolo_bbox), TRAIN_LABEL_DIR / f"{file_id}.txt"
            )
            cv2.imwrite(str(TRAIN_IMG_DIR / f"{file_id}.jpg"), waldo_img)
        else:
            write_text(format_yolo_string(yolo_bbox), VAL_LABEL_DIR / f"{file_id}.txt")
            cv2.imwrite(str(VAL_IMG_DIR / f"{file_id}.jpg"), waldo_img)
