# Potential help for best "least" colorful
## https://www.pyimagesearch.com/2017/06/05/computing-image-colorfulness-with-opencv-and-python/ 


# import the necessary packages
import format_iq_imgs as fii
import numpy as np
import cv2 as cv
from pathlib import Path
import os
from PIL import Image
from imutils import build_montages
from imutils import paths
import argparse
import imutils

os.getcwd()
IMG_DIR = Path("../../example-data/iq-test/dmi-api-test")
IMG_DIR.exists()

img_files = list(IMG_DIR.glob("*image*.png"))
test_img = fii.read_img(img_files[8])
image_list = fii.split_img(test_img)
# Image to alter
image = image_list[0][0]
image.shape
# Define Colors
## define the list of boundaries
boundaries = [
	([0, 0, 50], [60, 60, 200])
]
## Function for colorfulness
def image_colorfulness(image):
    # split the image into its respective RGB components
	(B, G, R) = cv.split(image.astype("float"))
	# compute rg = R - G
	rg = np.absolute(R - G)
	# compute yb = 0.5 * (R + G) - B
	yb = np.absolute(0.5 * (R + G) - B)
	# compute the mean and standard deviation of both `rg` and `yb`
	(rbMean, rbStd) = (np.mean(rg), np.std(rg))
	(ybMean, ybStd) = (np.mean(yb), np.std(yb))
	# combine the mean and standard deviations
	stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
	meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
	# derive the "colorfulness" metric and return it
	return stdRoot + (0.3 * meanRoot)

# loop over the boundaries
for (lower, upper) in boundaries:
	# create NumPy arrays from the boundaries
	lower = np.array(lower, dtype = "uint8")
	upper = np.array(upper, dtype = "uint8")
	# find the colors within the specified boundaries and apply
	# the mask
	mask = cv.inRange(image, lower, upper)
	output = cv.bitwise_and(image, image, mask = mask)

#Print Color Value
C = image_colorfulness(output)
C
#Potentially print
cv.imshow("Image",output)
cv.waitKey()
cv.destroyAllWindows() 