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

cv2.imshow("Image",image_list[0][0])
cv2.waitKey()
cv2.destroyAllWindows() 

# Rotate images 45 degrees
img_rotate_45_clockwise = cv2.rotate(image_list[0][0], cv2.ROTATE_45_CLOCKWISE)
##fii.show_img(img_rotate_45_clockwise)

# Rotate images 90 degrees
img_rotate_90_clockwise = cv2.rotate(image_list[0][0], cv2.ROTATE_90_CLOCKWISE)
##fii.show_img(img_rotate_90_clockwise)

# Rotate images 180 degrees
img_rotate_180_clockwise = cv2.rotate(image_list[0][0], cv2.ROTATE_180_CLOCKWISE)
##fii.show_img(img_rotate_180_clockwise)

# Game of life
## Find red 