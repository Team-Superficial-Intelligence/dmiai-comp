import cv2
import generate_imgs
import numpy as np
import waldominator
import imutils
from PIL import Image
from pathlib import Path


IMG_PATH = Path("../example-data/wheres-waldo/images")
ANN_PATH = IMG_PATH / "annotations"


def crop_img(img, bbox):
    return img[bbox[1] : bbox[3], bbox[0] : bbox[2]]


def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def format_loc(loc, template):
    xmid = int(loc[0] + template.shape[1] / 2)
    ymid = int(loc[1] + template.shape[0] / 2)
    return (xmid, ymid)


def match_template(target_img: np.array, template: np.array):
    result = cv2.matchTemplate(target_img, template, cv2.TM_CCOEFF_NORMED)
    (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
    return maxVal, maxLoc


def create_bb_img(bbox):
    """create np zeros with bbox dimensions"""
    return np.zeros((bbox[3] - bbox[1], bbox[2] - bbox[0]), dtype=np.uint8)


def create_template_list():
    ann_dict = generate_imgs.create_ann_dict(list(ANN_PATH.glob("*xml")))
    all_imgs = [create_bb_img(bbox) for bbox in ann_dict.values()]
    for i, (img_name, bbox) in enumerate(ann_dict.items()):
        img = cv2.imread(str(IMG_PATH / img_name))
        img = to_gray(img)
        test_img = crop_img(img, bbox)
        # flip_img = cv2.flip(img.copy(), 1)
        all_imgs[i] += test_img
        # generate_imgs.show_img(cv2.flip(test_img, 1))
        # break
    return all_imgs


def template_search(target_img, img_list):
    locs = [None for _ in img_list]
    confs = [0 for _ in img_list]
    for i, template in enumerate(img_list):
        try:
            score, loc = match_template(target_img, template)
        except cv2.error:
            print(i)
            raise
        confs[i] = score
        locs[i] = loc
        if score > 0.98:
            break
    return locs, confs


def find_match(target_img, img_list):
    target_img = to_gray(np.array(target_img))
    locs, confs = template_search(target_img, img_list)
    best_idx = np.argmax(confs)
    if len(confs) == 0:
        return (750, 750)
    return format_loc(locs[best_idx], img_list[best_idx]), img_list[best_idx]


def plot_match(pred, target_img):
    circle_img = cv2.circle(
        target_img.copy(), pred, radius=20, color=(0, 0, 255), thickness=6
    )
    return circle_img


TEMPLATES = create_template_list()

if __name__ == "__main__":
    TEST_PATH = Path("./dmi_submitted")
    test_paths = list(TEST_PATH.glob("*.png"))
    test_imgs = [Image.open(pat).convert("RGB") for pat in TEST_PATH.glob("*.png")]

    for i, target_img in enumerate(test_imgs):
        pred, pred_img = find_match(target_img, TEMPLATES)
        cv2.imwrite(f"{i}_pred.png", plot_match(pred, np.array(target_img)))
