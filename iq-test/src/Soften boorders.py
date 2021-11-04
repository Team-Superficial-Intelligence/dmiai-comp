import format_iq_imgs as fii
import numpy as np
import cv2 as cv
from pathlib import Path
import os
from PIL import Image

os.getcwd()
IMG_DIR = Path("../../example-data/iq-test/dmi-api-test")
IMG_DIR.exists()

img_files = list(IMG_DIR.glob("*image*.png"))
test_img = fii.read_img(img_files[3])
image_list = fii.split_img(test_img)
## Circle
Circle = np.zeros([110,110,3],dtype=np.uint8)
Circle.fill(255) # numpy array! 
Circle= cv.circle(Circle,(55,55),30,(0,0,0),-1)
#Circle = Image.fromarray(Circle) #convert numpy array to image
# Image to alter
TestImg = image_list[3][0]
TestImg.shape
## Combine to "soften"
Combined = cv.bitwise_or(TestImg,Circle)