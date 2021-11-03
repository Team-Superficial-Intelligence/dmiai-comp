import format_iq_imgs as fii
import rotate_check as rc
import cv2
import numpy as np
from pathlib import Path
from skimage.metrics import structural_similarity as compare_ssim


def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def xor_func(img1, img2):
    return to_gray(cv2.bitwise_xor(img1, img2))


def bitand(src1, src2):
    return cv2.bitwise_and(src1, src2)


def deltaE(img1, img2, colorspace=cv2.COLOR_BGR2LAB):
    # Convert BGR images to specified colorspace
    img1 = cv2.cvtColor(img1, colorspace)
    img2 = cv2.cvtColor(img2, colorspace)
    # compute the Euclidean distance with pixels of two images
    return np.mean(np.sqrt(np.sum((img1 - img2) ** 2, axis=-1)) / 255.0)


def check_colours(img_list, choices):
    test_list = img_list[:3]
    final_imgs = img_list[3][:2]
    crit = (
        np.mean(
            [
                compare_ssim(bitand(lst[0], lst[1]), lst[2], multichannel=True)
                for lst in test_list
            ]
        )
        > 0.83
    )
    if crit:
        best_guess = bitand(final_imgs[0], final_imgs[1])
        return np.argmax(compare_ssim(best_guess, choice) for choice in choices)


img_paths = rc.find_img_files()
img_path = img_paths[8]
choice_paths = rc.find_img_choices(img_path)
choices = [fii.read_img(f) for f in choice_paths]
img = fii.read_img(img_path)
img_list = fii.split_img(img)
print(check_colours(img_list, choices))


fii.show_img(img)


source1 = img_list[3][0]
source2 = img_list[3][1]
bitstuff = cv2.bitwise_and(source1, source2)
[deltaE(bitstuff, choice) for choice in choices]
fii.show_img(bitstuff)


fii.show_img(xor_func(source1, source1))
cv2.COLORBGR2
fii.show_img(bitstuff)

kernel = np.ones((5, 5), np.uint8)
fii.show_img(cv2.dilate(choices[1], kernel, iterations=1))

for choice in choices:
    fii.show_img(choice)
fii.show_img(bitstuff)
target = img_list[0][2]

