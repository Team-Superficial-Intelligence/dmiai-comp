# import the necessary packages
import format_iq_imgs as fii
import numpy as np
import cv2 as cv
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
fii.show_img(image)

# Convert BGR to HSV
# hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# https://answers.opencv.org/question/229620/drawing-a-rectangle-around-the-red-color-region/

# red color boundaries [B, G, R]
lower = [0, 0, 50]
upper = [30, 30, 200]

lower = np.array(lower, dtype="uint8")
upper = np.array(upper, dtype="uint8")

mask = cv.inRange(image, lower, upper)
output = cv.bitwise_and(image, image, mask=mask)

kernel = np.ones((5, 5), np.uint8)
output = cv.dilate(output, kernel, iterations=3)

# Rotate Triangle
# grab the dimensions of the image and calculate the center of the
# image
(h, w) = output.shape[:2]
(cX, cY) = (w // 2, h // 2)
# rotate our output image by 144 degrees around the center of the image
M = cv.getRotationMatrix2D((cX, cY), 144, 1.0)
rotated = cv.warpAffine(output, M, (w, h))
fii.show_img(rotated)
