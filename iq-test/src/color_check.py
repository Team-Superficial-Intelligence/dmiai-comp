import format_iq_imgs as fii
import rotate_check as rc
import cv2
import numpy as np
from pathlib import Path
from skimage.metrics import structural_similarity as compare_ssim


def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


img_paths = rc.find_img_files()
img_path = img_paths[0]
choice_paths = rc.find_img_choices(img_path)
choices = [fii.read_img(f) for f in choice_paths]
img = fii.read_img(img_path)
img_list = fii.split_img(img)

fii.show_img(img)

source1 = img_list[0][0]
source2 = img_list[0][1]
bitstuff = cv2.cvtColor(cv2.bitwise_xor(source1, source2), cv2.COLOR_BGR2GRAY)
gray_choices = [to_gray(choice) for choice in choices]


fii.show_img(bitstuff)
target = img_list[0][2]

