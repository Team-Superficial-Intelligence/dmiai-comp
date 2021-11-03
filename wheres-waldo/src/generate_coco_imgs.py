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
import imgaug as ia
import imgaug.augmenters as iaa

TRAIN_DIR = Path("../../example-data/wheres-waldo")
IMG_DIR = TRAIN_DIR / "images"
ANN_DIR = IMG_DIR / "annotations"
DATA_DIR = Path.home() / "datasets1" / "waldo"
assert DATA_DIR.exists()
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


def format_yolo_bbox_aug(bbox: List[int], img_dims=(300, 300)) -> List[float]:
    x_mid = ((bbox.x1 + bbox.x2) / 2) / img_dims[0]
    y_mid = ((bbox.y1 + bbox.y2) / 2) / img_dims[1]
    width = (bbox.x2 - bbox.x1) / img_dims[0]
    height = (bbox.y2 - bbox.y1) / img_dims[1]
    return [x_mid, y_mid, width, height]


def format_yolo_string(bbox: List[float]) -> str:
    full_dat = ["0"] + bbox
    return "\t".join(str(x) for x in full_dat)


def write_text(s: str, file_path: Path):
    with open(file_path, "w") as f:
        f.write(s)


def augmentation_pipeline():
    return iaa.Sequential([iaa.Fliplr(0.5), iaa.GaussianBlur((0, 2.5))],)


def move_to_val(fileid):
    old_img_name = TRAIN_IMG_DIR / f"{fileid}.jpg"
    new_img_name = VAL_IMG_DIR / f"{fileid}.jpg"
    old_lab_name = TRAIN_LABEL_DIR / f"{fileid}.txt"
    new_lab_name = VAL_LABEL_DIR / f"{fileid}.txt"
    old_img_name.rename(new_img_name)
    old_lab_name.rename(new_lab_name)


if __name__ == "__main__":
    annotations = list(ANN_DIR.glob("*.xml"))
    ann_dict = create_ann_dict(annotations)

    seq = augmentation_pipeline()
    N = 100  # Augs per img
    train_paths = list(ann_dict.keys())
    train_paths = [x for x in train_paths if x != "21.jpg"]

    train_dict = [
        create_waldo_crop(img_name, ann_dict)
        for img_name in train_paths
        for i in range(5)
    ]
    train_imgs = [x[0] for x in train_dict]
    train_boxes = [
        [ia.BoundingBox(x1=box[1][0], y1=box[1][1], x2=box[1][2], y2=box[1][3])]
        for box in train_dict
    ]

    aug_imgs, aug_bbs = seq(images=train_imgs * N, bounding_boxes=train_boxes * N)
    proper_bbs = [format_yolo_bbox_aug(aug_bb[0]) for aug_bb in aug_bbs]

    print("writing augmented imgs!")
    for img, bbox in zip(aug_imgs, proper_bbs):
        random_name = create_random_id()
        label_path = TRAIN_LABEL_DIR / f"{random_name}.txt"
        img_path = TRAIN_IMG_DIR / f"{random_name}.jpg"
        cv2.imwrite(str(img_path), img)
        write_text(format_yolo_string(bbox), label_path)

    print("writing normal imgs!")
    for img, bbox in zip(train_imgs, train_boxes):
        random_name = create_random_id()
        proper_box = format_yolo_bbox_aug(bbox[0])
        label_path = TRAIN_LABEL_DIR / f"{random_name}.txt"
        img_path = TRAIN_IMG_DIR / f"{random_name}.jpg"
        cv2.imwrite(str(img_path), img)
        write_text(format_yolo_string(proper_box), label_path)

    print("generating empty imgs!")
    for i in range(1000):
        random_img = random.choice(list(ann_dict.keys()))
        random_name = create_random_id()
        img = create_empty_crop(random_img, ann_dict)
        img_path = TRAIN_IMG_DIR / f"{random_name}.jpg"
        cv2.imwrite(str(img_path), img)

    # Finally, create validation set (randomly move 500)
    all_ids = list(TRAIN_IMG_DIR.glob("*.jpg"))
    all_ids = [file.name[:-4] for file in all_ids]
    val_ids = random.choices(all_ids, k=500)
    for val_id in val_ids:
        try:
            move_to_val(val_id)
        except FileNotFoundError:
            continue
