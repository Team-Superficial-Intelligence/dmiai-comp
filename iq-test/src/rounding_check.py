import format_iq_imgs as fii
import numpy as np
import cv2 as cv
import rotate_check as rc
from pathlib import Path
import os
from PIL import Image
from sklearn.cluster import MiniBatchKMeans

from skimage.metrics import structural_similarity as compare_ssim

def make_circle(size=30):
    Circle = np.zeros([110,110,3],dtype=np.uint8)
    Circle.fill(255) # numpy array! 
    Circle= cv.circle(Circle,(55,55),size,(0,0,0),-1)
    return Circle

def soften_func(img1, Circle):
    return cv.bitwise_or(img1,Circle)

def check_bitor(img_list, choices):
    Circle = make_circle(30)
    test_list = img_list[:3]
    final_imgs = img_list[3][:2]
    crit = (
        np.mean(
            [
                compare_ssim(soften_func(lst[0], Circle), lst[2], multichannel=True)
                for lst in test_list
            ]
        )
        > 0.70
    )
    if crit:
        best_guess = bitor(final_imgs[0], final_imgs[1])
        return np.argmax(compare_ssim(best_guess, choice) for choice in choices)
    return None








############################ EXTRA STUFF #################################
#os.getcwd()
#IMG_DIR = Path("../../example-data/iq-test/dmi-api-test")
#IMG_DIR.exists()

#img_files = list(IMG_DIR.glob("*image*.png"))
#test_img = fii.read_img(img_files[3])
#image_list = fii.split_img(test_img)
## Circle
#Circle = np.zeros([110,110,3],dtype=np.uint8)
#Circle.fill(255) # numpy array! 
#Circle= cv.circle(Circle,(55,55),30,(0,0,0),-1)
#Circle = Image.fromarray(Circle) #convert numpy array to image
# Image to alter
#TestImg = image_list[3][0]
#TestImg.shape
## Combine to "soften"
#Combined = cv.bitwise_or(TestImg,Circle)