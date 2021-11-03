import format_iq_imgs as fii
import cv2
import numpy as np
import re
from skimage.metrics import structural_similarity as compare_ssim
from pathlib import Path


def find_identifier(img_path):
    img_name = img_path.name
    return re.match(r"rq_\d+", img_name).group(0)


def check90clockwise(source_img, target_img):
    rotate_img = cv2.rotate(source_img, cv2.ROTATE_90_CLOCKWISE)
    sim_score = compare_ssim(rotate_img, target_img, multichannel=True)
    if sim_score > 0.9:
        return True


def check90counterclockwise(source_img, target_img):
    rotate_img = cv2.rotate(source_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    sim_score = compare_ssim(rotate_img, target_img, multichannel=True)
    if sim_score > 0.9:
        return True


def check180(source_img, target_img):
    rotate_img = cv2.rotate(source_img, cv2.ROTATE_180)
    sim_score = compare_ssim(rotate_img, target_img, multichannel=True)
    if sim_score > 0.9:
        return True


def rotate90clockwise(img):
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)


def rotate90counterclockwise(img):
    return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)


def rotate180(img):
    return cv2.rotate(img, cv2.ROTATE_180)


def check_similarity(source_img, target_img, func):
    new_img = func(source_img)
    return compare_ssim(new_img, target_img, multichannel=True) > 0.9


def find_best_match(source_img, choices, func):
    image_guess = func(source_img)
    sim_scores = [
        compare_ssim(image_guess, choice, multichannel=True) for choice in choices
    ]
    return np.argmax(sim_scores)


def check_rotations(full_list, choices):
    test_cases = full_list[:3]
    final_img = full_list[3][0]
    # Check 90 Counter
    case1 = all(
        check_similarity(lst[0], lst[2], rotate90counterclockwise) for lst in test_cases
    )
    if case1:
        return find_best_match(final_img, choices, rotate90counterclockwise)
    # Check 90
    case2 = all(
        check_similarity(lst[0], lst[2], rotate90clockwise) for lst in test_cases
    )
    if case2:
        return find_best_match(final_img, choices, rotate90clockwise)
    case3 = all(check_similarity(lst[0], lst[2], rotate180) for lst in test_cases)
    if case3:
        return find_best_match(final_img, choices, rotate180)
    return None


IMG_PATH = Path("../../example-data/iq-test/dmi-api-test")

img_files = list(IMG_PATH.glob("*image*.png"))

img_path = img_files[2]
img_choices = list(IMG_PATH.glob(f"*{find_identifier(img_path)}*choice*"))

choices = [fii.read_img(f) for f in img_choices]

img = fii.read_img(img_path)
fii.show_img(img)

img_list = fii.split_img(img)
winner_idx = check_rotations(img_list, choices)

fii.show_img(choices[winner_idx])
