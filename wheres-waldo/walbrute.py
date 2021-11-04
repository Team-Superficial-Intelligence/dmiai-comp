import cv2
import generate_imgs
import numpy as np
import waldominator
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


def match_template(target_img, template):
    result = cv2.matchTemplate(
        to_gray(target_img), to_gray(template), cv2.TM_CCOEFF_NORMED
    )
    (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
    return maxVal, maxLoc


ann_dict = generate_imgs.create_ann_dict(list(ANN_PATH.glob("*xml")))

all_imgs = []
for img_name, bbox in ann_dict.items():
    img = cv2.imread(str(IMG_PATH / img_name))
    test_img = crop_img(img, bbox)
    # flip_img = cv2.flip(img.copy(), 1)
    img_list = [test_img]
    all_imgs.extend(img_list)
    # generate_imgs.show_img(cv2.flip(test_img, 1))
    # break


test_path = IMG_PATH.parent / "waldo.jpg"
test_img = cv2.imread(str(test_path))


def find_match(target_img, img_list):
    locs = [None for _ in img_list]
    confs = [0 for _ in img_list]
    for i, template in enumerate(img_list):
        score, loc = match_template(target_img, template)
        confs[i] = score
        locs[i] = loc
        if score > 0.98:
            break
    best_idx = np.argmax(confs)
    return format_loc(locs[best_idx], img_list[best_idx])


preds = find_match(test_img, all_imgs)
circle = cv2.circle(test_img.copy(), preds, radius=5, color=(0, 255, 0), thickness=5)


test_img, bbox = generate_imgs.create_training_img(ann_dict)
pred = find_match(test_img, all_imgs)
pred

N = 10
score = 0
for i in range(N):
    test_img, bbox = generate_imgs.create_training_img(ann_dict)
    pred = find_match(test_img, all_imgs)
    score += waldominator.eval_pred(pred[0], pred[1], bbox)
waldominator.eval_pred(pred[0], pred[1], bbox)
cv2.imwrite("test_img.png", circle)


img_list, _ = waldominator.split_img_matrix(test_img)

