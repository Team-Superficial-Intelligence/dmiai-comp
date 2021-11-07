# import the necessary packages
import shit_checker.format_iq_imgs as fii
import shit_checker.rotate_check as rc
import numpy as np
import cv2
from skimage.metrics import structural_similarity as compare_ssim
from pathlib import Path
from typing import List
import os
from PIL import Image
from imutils import build_montages
from imutils import paths
import argparse
import imutils

import matplotlib.pyplot as plt

os.getcwd()
IMG_DIR = Path("../example-data/iq-test/dmi-api-test")
IMG_DIR.exists()

img_files = list(IMG_DIR.glob("*image*.png"))
test_img = fii.read_img(img_files[-3])
image_list = fii.split_img(test_img)

# Image to alter
image = image_list[0][0]
final_image = image_list[3][1]
fii.show_img(image)

# Convert BGR to HSV
# hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# https://answers.opencv.org/question/229620/drawing-a-rectangle-around-the-red-color-region/

# red color boundaries [B, G, R]
def pop_color(img, boundary):
    lower = np.array(boundary[0], dtype="uint8")
    upper = np.array(boundary[1], dtype="uint8")
    mask = cv2.inRange(img, lower, upper)
    output = cv2.bitwise_and(img, img, mask=mask)
    return output


def rotate_triangle(tri_img, degrees=144):
    # Rotate Triangle
    # grab the dimensions of the image and calculate the center of the
    # image
    (h, w) = tri_img.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # rotate our output image by 144 degrees around the center of the image
    M = cv2.getRotationMatrix2D((cX, cY), degrees, 1.0)
    return cv2.warpAffine(tri_img, M, (w, h))


def check_star(full_list, choices):
    red_boundaries = ([0, 0, 50], [30, 30, 200])
    boundaries = [red_boundaries]
    for boundary in boundaries:
        processed_choices = [pop_color(img, boundary) for img in choices]
        for degrees in (72, 144, 216, 288):

            return None
    return None


# red color boundaries [B, G, R]
lower = [0, 0, 140]
upper = [120, 100, 230]
red_boundaries = (lower, upper)
fii.show_img(final_image)
mask = cv2.inRange(
    final_image, np.array(red_boundaries[0]), np.array(red_boundaries[1])
)
mask = np.dstack((mask, mask, mask))

no_sky = cv2.bitwise_and(final_image, mask)

fii.show_img(no_sky)

# Blue color boundaries [B, G, R]
lower = [180, 160, 50]
upper = [210, 180, 85]


# Yellow color boundaries [B, G, R]
lower = [50, 120, 160]
upper = [100, 150, 210]

# green color boundaries [B, G, R]
lower = [40, 125, 60]
upper = [100, 170, 120]
boundary = red_boundaries

np.mean(
    [
        compare_ssim(
            rotate_triangle(pop_color(lst[1], boundary), degrees=72),
            pop_color(lst[2], boundary),
            multichannel=True,
        )
        for lst in image_list[:3]
    ]
)

fii.show_img(image_list[1][0])

tri_img = pop_color(final_image, red_boundaries)
pred = rotate_triangle(tri_img)

choices = [
    fii.read_img(choice)
    for choice in rc.find_img_choices(img_files[-3], img_dir=IMG_DIR)
]
processed_choices = [pop_color(img, red_boundaries) for img in choices]
sims = [compare_ssim(pred, choice, multichannel=True) for choice in processed_choices]
sims
fii.show_img(choices[np.argmax(sims)])

fii.show_img(image)
