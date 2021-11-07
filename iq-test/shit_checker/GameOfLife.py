# Code to count dots
import functools
import shit_checker.format_iq_imgs as fii
import shit_checker.color_check as cc
import shit_checker.rotate_check as rc
import numpy as np
import cv2
from pathlib import Path
from typing import List
import os
from skimage.metrics import structural_similarity as compare_ssim
import imutils

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

def cnt_size(cnt):
    _, _, w, _ = cv2.boundingRect(cnt)
    return w

def get_contour_precedence(contour, cols):
    tolerance_factor = 10
    origin = cv2.boundingRect(contour)
    return ((origin[1] // tolerance_factor) * tolerance_factor) * cols + origin[0]

def ultimate_mask(img):
    black = (np.array([0, 0, 0]),np.array([30, 30, 30]))
    boundaries = cc.BOUNDARIES + [black]
    masks = [cv2.inRange(img, bound[0], bound[1]) for bound in boundaries]
    return functools.reduce(cv2.bitwise_or, masks)

def get_cnts(img):
    imgray = cc.to_gray(img)
    _, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours= cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(contours)
    cnts = [cnt for cnt in cnts if cnt_size(cnt)<80]  
    cnts.sort(key=lambda x:get_contour_precedence(x, image.shape[1]))
    return cnts

def count_color(img, bound):
    return np.sum(cv2.inRange(img, bound[0], bound[1]))


def find_cnt_color(image, cnt):
    x, y, w, h = cv2.boundingRect(cnt)
    img_crop = image[y:y+h, x:x+w]
    bnd_count = [count_color(img_crop, bound) for bound in cc.BOUNDARIES]
    return np.argmax(bnd_count)

def cnt_to_nums(img, cnts):
    num_array = np.array([find_cnt_color(img, cnt) for cnt in cnts])
    unique_cols = np.unique(num_array)
    if len(unique_cols) == 2:
        num_array = num_array == unique_cols[0]
    return num_array

def convert_to_nums(img):
    cnts = get_cnts(img)
    return cnt_to_nums(img, cnts)

if __name__ == "__main__":
    # reading the image in grayscale mode
    IMG_DIR = Path("../example-data/iq-test/dmi-api-test")
    IMG_DIR.exists()

    img_files = list(IMG_DIR.glob("*image*.png"))

    img_path = img_files[0]
    test_img = fii.read_img(img_path)
    image_list = fii.split_img(test_img)

    choices = [
        fii.read_img(choice)
        for choice in rc.find_img_choices(img_path, img_dir=IMG_DIR)
    ]
    # Image to alter

    image = image_list[0][1]
    fii.show_img(image)
    a = convert_to_nums(image)

    shapeMask = ultimate_mask(image) 
    cnts = cv2.findContours(shapeMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = [cnt for cnt in cnts if cnt_size(cnt) < 100]
    cnts.sort(key=lambda x:get_contour_precedence(x, image.shape[1]))
    fii.show_img(shapeMask)
    for cnt in cnts:
        new_img = cv2.drawContours(image.copy(), [cnt], -1, (0, 255, 0), 2)
        fii.show_img(new_img)
	cv2.waitKey(0)
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
