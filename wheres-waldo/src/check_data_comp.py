"""
Checks whether we have the right dataset
"""
from pathlib import Path
import cv2
import numpy as np
import xml.etree.ElementTree as ET

ANN_PATH = Path("./extra_anns")
IMG_DIR = Path("../example-data/wheres-waldo/images")
ANN_IMGS = Path("./dmi_submitted")


def show_img(img):
    cv2.imshow("yes", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def match_template(target_img: np.array, template: np.array):
    result = cv2.matchTemplate(target_img, template, cv2.TM_CCOEFF_NORMED)
    (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
    return maxVal, maxLoc


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
    return {filename: list_with_all_boxes}


def read_img(img_path):
    return cv2.imread(str(img_path))


def crop_img(img, bbox):
    return img[bbox[1] : bbox[3], bbox[0] : bbox[2]]


def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


ann_list = [read_content(xml_file) for xml_file in ANN_PATH.glob("*.xml")]
img_paths = list(IMG_DIR.glob("*.*g"))
imgs = [to_gray(read_img(img_path)) for img_path in img_paths]

crop_list = []
for dic in ann_list:
    img_name = list(dic.keys())[0]
    img = read_img(ANN_IMGS / img_name)
    crop_list.extend([to_gray(crop_img(img, bbox)) for bbox in dic[img_name]])

imgs = imgs[1:]
for j, template in enumerate(crop_list):
    for i, img in enumerate(imgs):
        top, loc = match_template(img, template)
        if top > 0.9:
            print(f"succes at {i} for template {j}")
            break


test_img = read_img(next(IMG_DIR.glob("*pirate2*")))

test_img = test_img.shape

match_template(to_gray(test_img), crop_list[3])

show_img(crop_list[3])
show_img(to_gray(test_img))

crop_list[3].shape
