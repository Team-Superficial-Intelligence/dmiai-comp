import shit_checker.format_iq_imgs as fii
import cv2
import numpy as np
import re
from skimage.metrics import structural_similarity as compare_ssim
from pathlib import Path
from typing import List

IMG_PATH = Path("../../example-data/iq-test/dmi-api-test")


def find_identifier(img_path):
    img_name = img_path.name
    return re.match(r"rq_\d+", img_name).group(0)


def rotate90clockwise(img):
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)


def rotate90counterclockwise(img):
    return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)


def rotate180(img):
    return cv2.rotate(img, cv2.ROTATE_180)


def check_similarity(source_img, target_img, func):
    new_img = func(source_img)
    return compare_ssim(new_img, target_img, multichannel=True) > 0.87


def find_best_match(source_img, choices, func):
    image_guess = func(source_img)
    sim_scores = [
        compare_ssim(image_guess, choice, multichannel=True) for choice in choices
    ]
    return np.argmax(sim_scores)


def check_rotation(full_list, choices, func):
    test_cases = full_list[:3]
    final_img = full_list[3][0]
    # Check 90 Counter
    case = all(check_similarity(lst[0], lst[2], func) for lst in test_cases)
    if case:
        return find_best_match(final_img, choices, func)
    return None


def check_rotations(full_list, choices):
    func_list = [rotate90clockwise, rotate90counterclockwise, rotate180]
    for func in func_list:
        result = check_rotation(full_list, choices, func)
        if result:
            return result
    return None


def find_img_choices(img_path: Path, img_dir=None) -> List[Path]:
    identifier = find_identifier(img_path)
    if img_dir is None:
        img_dir = IMG_PATH
    return list(img_dir.glob(f"*{identifier}*choice*.png"))


def find_img_files(img_path=None):
    if img_path is None:
        img_path = IMG_PATH
    return list(img_path.glob("*image*.png"))


if __name__ == "__main__":

    img_files = list(IMG_PATH.glob("*image*.png"))

    img_path = img_files[4]
    img_choices = find_img_choices(img_path)

    choices = [fii.read_img(f) for f in img_choices]

    img = fii.read_img(img_path)
    fii.show_img(img)

    img_list = fii.split_img(img)
    full_list = img_list
    winner_idx = check_rotations(img_list, choices)

    fii.show_img(choices[winner_idx])
