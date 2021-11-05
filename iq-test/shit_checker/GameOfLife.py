# Code to count dots
import format_iq_imgs as fii
import numpy as np
import cv2 as cv
from pathlib import Path
import re
from typing import List
import os
from PIL import Image
from skimage.metrics import structural_similarity
from imutils import build_montages
from imutils import paths
import argparse
import imutils
import matplotlib.pyplot as plt

# reading the image in grayscale mode
os.getcwd()
IMG_DIR = Path("../../example-data/iq-test/dmi-api-test")
IMG_PATH = Path("../../example-data/iq-test/dmi-api-test")
IMG_DIR.exists()
IMG_PATH.exists()

img_files = list(IMG_DIR.glob("*image*.png"))
test_img = fii.read_img(img_files[2])
image_list = fii.split_img(test_img)
# Image to alter
image = image_list[0][1]
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
## Check
#cv.imshow('diff.png', gray)
#cv.waitKey()
#cv.destroyAllWindows() 


# threshold
th, threshed = cv.threshold(gray, 100, 255,
		cv.THRESH_BINARY|cv.THRESH_OTSU)

# findcontours
cnts = cv.findContours(threshed, cv.RETR_LIST,
					cv.CHAIN_APPROX_SIMPLE)[-2]

# filter by area
s1 = 1
s2 = 100
xcnts = []

for cnt in cnts:
	if s1<cv.contourArea(cnt) <s2:
		xcnts.append(cnt)

# printing output
print("\nDots number: {}".format(len(xcnts)))

# Above 4


# Intuition
## Look at picture 2 
## Pick Picture 3 that is furthest apart from one color distribution

# import the necessary packages
import format_iq_imgs as fii
import numpy as np
import cv as cv
from pathlib import Path
import re
from typing import List
import os
from PIL import Image
from skimage.metrics import structural_similarity
from imutils import build_montages
from imutils import paths
import argparse
import imutils
import matplotlib.pyplot as plt

os.getcwd()
IMG_DIR = Path("../../example-data/iq-test/dmi-api-test")
IMG_PATH = Path("../../example-data/iq-test/dmi-api-test")
IMG_DIR.exists()
IMG_PATH.exists()

img_files = list(IMG_DIR.glob("*image*.png"))
test_img = fii.read_img(img_files[2])
image_list = fii.split_img(test_img)
# Image to alter
image = image_list[0][1]

# Testimage
def find_identifier(img_path):
    img_name = img_path.name
    return re.match(r"rq_\d+", img_name).group(0)

def find_img_choices(img_path: Path) -> List[Path]:
    identifier = find_identifier(img_path)
    return list(IMG_PATH.glob(f"*{identifier}*choice*.png"))

img_files = list(IMG_PATH.glob("*image*.png"))
img_path = img_files[4]
img_choices = find_img_choices(img_path)
choices = [fii.read_img(f) for f in img_choices]

# Compute Difference using Sickit
(score, diff) = structural_similarity(image, choices[0], full=True,multichannel=True)
print("Image similarity", score)


# Print
# cv.imwrite('diff.png', difference)
# Potentially show
cv.imshow('diff.png', difference)
cv.waitKey()
cv.destroyAllWindows() 






# Leftovers
# Convert BGR to HSV
#hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
# https://answers.opencv.org/question/229620/drawing-a-rectangle-around-the-red-color-region/

# red color boundaries [B, G, R]
## define the list of boundaries
boundaries = [
    ([0, 0, 50], [60, 60, 200])
]

for (lower, upper) in boundaries:
    # create NumPy arrays from the boundaries
    lower = np.array(lower, dtype = "uint8")
    upper = np.array(upper, dtype = "uint8")
    # find the colors within the specified boundaries and apply the mask
    # For the second image we have
    mask = cv.inRange(image, lower, upper)
    output_we_know = cv.bitwise_and(image, image, mask = mask)
    # For testimage
    mask = cv.inRange(testimage, lower, upper)



# Testing
##cv.imshow('diff.png', testimage)
##cv.waitKey()
##cv.destroyAllWindows() 

#how
## Remove everything but one color
## Add picture 2 on top of picture 3 in so they cancel each other out
## Look at what's left potential_choice_output= cv.bitwise_and(testimage, image, mask = mask)
