# Code to count dots
import shit_checker.format_iq_imgs as fii
import shit_checker.color_check as cc
import shit_checker.rotate_check as rc
import numpy as np
import cv2
from pathlib import Path
import re
from typing import List
import os
from PIL import Image
from skimage.metrics import structural_similarity as compare_ssim
from imutils import build_montages
from imutils import paths
import argparse
import imutils
import matplotlib.pyplot as plt


def count_dots(img):
    gray = cc.to_gray(img)
    # threshold
    _, threshed = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # findcontours
    cnts = cv2.findContours(threshed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
    # filter by area
    s1 = 1
    s2 = 500
    return sum((s1 < cv2.contourArea(cnt) < s2) for cnt in cnts)


def check_grid(full_list, choices):
    final_img = full_list[3][1]
    is_grid = mean_dots(full_list) > 15
    if is_grid:
        choice_sim = [
            compare_ssim(final_img, choice, multichannel=True) for choice in choices
        ]
        return np.argmin(choice_sim)
    return None


def mean_dots(full_list):
    return np.mean([count_dots(img) for lst in full_list for img in lst])


if __name__ == "__main__":
    # reading the image in grayscale mode
    os.getcwd()
    IMG_DIR = Path("../example-data/iq-test/dmi-api-test")
    IMG_DIR.exists()

    img_files = list(IMG_DIR.glob("*image*.png"))

    test_img = fii.read_img(img_files[4])
    image_list = fii.split_img(test_img)

    img_path = img_files[4]
    choices = [
        fii.read_img(choice)
        for choice in rc.find_img_choices(img_path, img_dir=IMG_DIR)
    ]
    image_list = fii.split_img(fii.read_img(img_path))
    # Image to alter

    image = image_list[0][1]
    fii.show_img(image)

    count_dots(image)
    full_list = image_list
    check_grid(image_list, choices)

    for img_path in img_files:
        img = fii.read_img(img_path)
        image_list = fii.split_img(img)
        choices = [
            fii.read_img(choice)
            for choice in rc.find_img_choices(img_path, img_dir=IMG_DIR)
        ]
        # print(mean_dots(image_list))
        print(check_grid(image_list, choices))
