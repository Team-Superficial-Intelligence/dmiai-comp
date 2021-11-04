import cv2
import generate_imgs
import numpy as np
import waldominator
import imutils
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


def create_bb_img(bbox):
    """create np zeros with bbox dimensions"""
    return np.zeros((bbox[3] - bbox[1], bbox[2] - bbox[0], 3), dtype=np.uint8)


def create_template_list():
    ann_dict = generate_imgs.create_ann_dict(list(ANN_PATH.glob("*xml")))
    all_imgs = [create_bb_img(bbox) for bbox in ann_dict.values()]
    for i, (img_name, bbox) in enumerate(ann_dict.items()):
        img = cv2.imread(str(IMG_PATH / img_name))
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
    locs, confs = template_search(target_img, img_list)
    best_idx = np.argmax(confs)
    return format_loc(locs[best_idx], img_list[best_idx]), img_list[best_idx]


def plot_match(pred, target_img):
    circle_img = cv2.circle(
        target_img.copy(), pred, radius=20, color=(0, 0, 255), thickness=6
    )
    return circle_img


TEMPLATES = create_template_list()

if __name__ == "__main__":

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

    TEST_PATH = Path("./dmi_submitted")
    test_paths = list(TEST_PATH.glob("*.png"))
    test_imgs = [cv2.imread(str(pat)) for pat in TEST_PATH.glob("*.png")]

    for i, target_img in enumerate(test_imgs):
        pred, pred_img = find_match(target_img, all_imgs)
        cv2.imwrite(f"{i}_pred.png", plot_match(pred, target_img))

    target_img = test_imgs[15]

    _, pred_img = find_match(test_imgs[13], all_imgs)

    generate_imgs.show_img(pred_img)
    tempfind_matchtest_imgs[13]

    template = all_imgs[28]

    for i, img in enumerate(all_imgs):
        try:
            to_gray(img)
        except cv2.error:
            print(i)

    confs, locs = template_search(target_img, all_imgs)
    generate_imgs.show_img

    gray = to_gray(target_img)
    # template = cv2.Canny(template, 50, 200)
    (tH, tW) = template.shape[:2]
    found = None
    for scale in np.linspace(0.2, 1.0, 20)[::-1]:
        # resize the image according to the scale, and keep track
        # of the ratio of the resizing
        resized = imutils.resize(gray, width=int(gray.shape[1] * scale))
        r = gray.shape[1] / float(resized.shape[1])
        # if the resized image is smaller than the template, then break
        # from the loop
        if resized.shape[0] < tH or resized.shape[1] < tW:
            break
        # detect edges in the resized, grayscale image and apply template
        # matching to find the template in the image
        # edged = cv2.Canny(resized, 50, 200)
        result = cv2.matchTemplate(resized, template, cv2.TM_CCOEFF)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

        # if we have found a new maximum correlation value, then update
        # the bookkeeping variable
        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r)

    (_, maxLoc, r) = found
    (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
    (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

    match_template(target_img, template)

    found[0]
    generate_imgs.show_img(target_img[startX:endX, startY:endY])

    target_img = test_imgs[1]

    locs, confs = template_search(target_img, all_imgs)
    pred, pred_img = find_match(target_img, all_imgs)
    confs
    generate_imgs.show_img(pred_img)
    pred

    confs[28]
    generate_imgs.show_img(all_imgs[28])

    len(confs)

    for i, img in enumerate(all_imgs):
        cv2.imshow(f"{i}", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
