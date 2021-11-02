"""
Generate training data for Waldo. 
Dimensions are initially 1500 x 1500 composed of 300x300 tiles with one tile having waldo in it 
"""
from typing import Dict, List
from numpy.random.mtrand import random
import random
import cv2
import numpy as np
import json
import xml.etree.ElementTree as ET
import itertools
from pathlib import Path

TRAIN_DIR = Path("../../example-data/wheres-waldo")
IMG_DIR = TRAIN_DIR / "images"
ANN_DIR = IMG_DIR / "annotations"


def get_img_path(img_name: str) -> str:
    return str(IMG_DIR / img_name)


def show_img(img: np.ndarray):
    """ Helper function for showing images """
    cv2.imshow("Show", img)
    cv2.waitKey()
    cv2.destroyAllWindows()


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


def draw_bbox(img, bbox) -> np.ndarray:
    """ Helper function for drawing a bounding box """
    new_img = img.copy()
    return cv2.rectangle(
        new_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2
    )


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


def create_ann_dict(annotations: List[Path]) -> Dict[str, List[int]]:

    ann_list = [[None, None] for _ in annotations]
    for i, annotation in enumerate(annotations):
        file_name, bboxes = read_content(annotation)
        ann_list[i][0] = file_name
        ann_list[i][1] = bboxes
    return {item[0]: item[1] for item in ann_list}


def create_training_img(ann_dict, shape=(5, 5), crop_shape=(300, 300)):
    w = shape[0]
    h = shape[1]
    mat_x = w * crop_shape[0]
    mat_y = h * crop_shape[1]
    imgmatrix = np.zeros((mat_x, mat_y, 3), np.uint8)
    waldo_selection = list(ann_dict.keys())
    waldo_selection.remove("21.jpg")  # Weird edge case
    waldo_img, waldo_bbox = create_waldo_crop(
        random.choice(waldo_selection), ann_dict, img_dims=crop_shape
    )
    num_random_imgs = shape[0] * shape[1] - 1
    random_imgs = [
        (0, create_empty_crop(img_path, ann_dict, img_dims=crop_shape))
        for img_path in random.choices(list(ann_dict.keys()), k=num_random_imgs)
    ]
    positions = list(itertools.product(range(5), range(5)))
    random_imgs.append((1, waldo_img))
    random.shuffle(random_imgs)

    waldo_idx = [item[0] for item in random_imgs].index(1)
    imgs = [img[1] for img in random_imgs]
    i = 0
    for (y_i, x_i), img in zip(positions, imgs):
        x = x_i * crop_shape[0]
        y = y_i * crop_shape[1]
        imgmatrix[y : y + crop_shape[0], x : x + crop_shape[1], :] = img
        if i == waldo_idx:
            waldo_bbox[0] += x
            waldo_bbox[1] += y
            waldo_bbox[2] += x
            waldo_bbox[3] += y
        i += 1
    return imgmatrix, waldo_bbox


def dump_json(obj, file_path):
    with open(file_path, "w") as f:
        json.dump(obj, f)


annotations = list(ANN_DIR.glob("*.xml"))
ann_dict = create_ann_dict(annotations)
test_img = cv2.imread(get_img_path("1.jpg"))

training_img, waldo_bbox = create_training_img(ann_dict)
# show_img(training_img)
new_img = draw_bbox(training_img, waldo_bbox)
show_img(new_img)

img_dir = TRAIN_DIR / "training_imgs"
N = 100
for i in range(N):
    print(f"creating img {i+1} of {N}")
    training_img, waldo_bbox = create_training_img(ann_dict)

    print("writing data...")
    file_path = img_dir / f"training_{i}.png"
    bbox_dict = {file_path.name: waldo_bbox}
    cv2.imwrite(str(file_path), training_img)
    dump_json(bbox_dict, img_dir / "bboxes" / f"training_{i}.json")
