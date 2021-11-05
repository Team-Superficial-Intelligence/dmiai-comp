# import the necessary packages
import format_iq_imgs as fii
import numpy as np
import cv2 as cv
from pathlib import Path
import os
from PIL import Image
from imutils import build_montages
from imutils import paths
import imutils

# Define Colors

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
def Pop_Out_Function(input_img):
    ## define the list of boundaries
    boundaries = [([0, 0, 50], [60, 60, 200])]
    for (lower, upper) in boundaries:
        # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        # find the colors within the specified boundaries and apply the mask
        mask = cv.inRange(image, lower, upper)
        output = cv.bitwise_and(image, image, mask=mask)
        return output


## Source for this: https://www.pyimagesearch.com/2017/06/05/computing-image-colorfulness-with-opencv-and-python/
def Check_Least_Colorful_Function(img_list):
    # initialize the results list
    results = []
    # loop over the image paths
    for image in img_list:
        # Kick out everything but the red dots
        Red_dots = Pop_Out_Function(image)
        # compute the colorfulness metric for the image
        Colorfulness = image_colorfulness(Red_dots)
        # add the image and colorfulness metric to the results list
        results.append((image, Colorfulness))
    # sort the results with more colorful images at the front of the list
    results = sorted(results, key=lambda x: x[1], reverse=True)
    # build the lists of the *least colorful* images
    leastColor = [r[0] for r in results[-1:]][::-1]
    return leastColor


############################ EXTRA STUFF #################################
# Path and loading
# os.getcwd()
# IMG_DIR = Path("../../example-data/iq-test/dmi-api-test")
# IMG_DIR.exists()

# img_files = list(IMG_DIR.glob("*image*.png"))
# test_img = fii.read_img(img_files[8])
# image_list = fii.split_img(test_img)
# Image to alter
# image = image_list[0][0]
# image.shape

# Potentially print
# cv.imshow("Image",output)
# cv.waitKey()
# cv.destroyAllWindows()

