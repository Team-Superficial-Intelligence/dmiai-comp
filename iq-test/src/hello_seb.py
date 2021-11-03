import format_iq_imgs as fii
import cv2
from pathlib import Path
import os

os.getcwd()
IMG_DIR = Path("../../example-data/iq-test/dmi-api-test")
IMG_DIR.exists()

img_files = list(IMG_DIR.glob("*image*.png"))
test_img = fii.read_img(img_files[0])

image_list = fii.split_img(test_img)

fii.show_img(image_list[0][0])

print(image_list[0][0].shape)
# (225, 400, 3)
cv2.rotate(image_list[0][0])
