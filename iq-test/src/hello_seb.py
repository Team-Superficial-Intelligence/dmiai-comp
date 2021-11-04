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

import matplotlib.pyplot as plt

os.getcwd()
IMG_DIR = Path("../../example-data/iq-test/dmi-api-test")
IMG_DIR.exists()

img_files = list(IMG_DIR.glob("*image*.png"))
test_img = fii.read_img(img_files[0])
image_list = fii.split_img(test_img)
# Image to alter
image = image_list[0][0]

#Potentially print
cv.imshow("Image",image)
cv.waitKey()
cv.destroyAllWindows() 


## Softness
circle = cv.circle(blank.copy(),(200,200),200,255,-1)
small_circlecv.circle(blank.copy(),(200,200),100,255,-1)
Triangle = cv.triangle(blank.copy(),(200,200),200,255,-1)
Triangle = cv.rectangle(blank.copy(),(30,30),(370,370),255,-1)
## Combine to soften
bitwise_and = cv.bitwise_and(img, circle)


#Addition
Pic1 = image_list[0][0]
Pic2 = image_list[0][1]
### Bitwise
bitwise_addition = cv.bitwise_and(Pic1, Pic2)
### Greyscale
bitwise_addition = cv.cvtColor(bitwise_addition, cv.COLOR_RGBA2GRAY)
### Print
cv.imshow("Bitwise_Addition",bitwise_addition)
cv.waitKey()
cv.destroyAllWindows() 
### Erode
kernel = np.ones((5,5),np.uint8)
Pic1 = cv.erode(Pic1,kernel,iterations=3)
Pic2 = cv.erode(Pic2,kernel,iterations=3)
### BLUR
Pi1 = cv.GaussianBlur(Pic1,(7,7),cv.BORDER_DEFAULT)
Pic2 = cv.GaussianBlur(Pic2,(7,7),cv.BORDER_DEFAULT)
### Greyscale
Pic1 = cv.cvtColor(Pic1, cv.COLOR_RGBA2GRAY)
Pic2 = cv.cvtColor(Pic2, cv.COLOR_RGBA2GRAY)
### Edge cascade
blur = cv.GaussianBlur(img,(7,7),cv.BORDER_DEFAULT)
canny = cv.Canny(blur,125,175

## Softness
circle = cv.circle(blank.copy(),(200,200),200,255,-1)
small_circlecv.circle(blank.copy(),(200,200),100,255,-1)
Triangle = cv.triangle(blank.copy(),(200,200),200,255,-1)
Triangle = cv.rectangle(blank.copy(),(30,30),(370,370),255,-1)
## Combine to soften
bitwise_and = cv.bitwise_and(img, circle)


# Rotate images 45 degrees
img_rotate_45_clockwise = cv.rotate(image_list[0][0], cv.ROTATE_45_CLOCKWISE)
##fii.show_img(img_rotate_45_clockwise)

# Rotate images 90 degrees
img_rotate_90_clockwise = cv.rotate(image_list[0][0], cv.ROTATE_90_CLOCKWISE)
##fii.show_img(img_rotate_90_clockwise)

# Rotate images 180 degrees
img_rotate_180_clockwise = cv.rotate(image_list[0][0], cv.ROTATE_180_CLOCKWISE)
##fii.show_img(img_rotate_180_clockwise)


# Show One picture
SigleImageLookup = image_list[0][1]
cv.imshow("Image",SigleImageLookup)
cv.waitKey()
cv.destroyAllWindows() 










# Done
##Cross
import format_iq_imgs as fii
import numpy as np
import cv
from pathlib import Path
import os

os.getcwd()
IMG_DIR = Path("../../example-data/iq-test/dmi-api-test")
IMG_DIR.exists()

img_files = list(IMG_DIR.glob("*image*.png"))
test_img = fii.read_img(img_files[6])

image_list = fii.split_img(test_img)

#Addition
Pic1 = image_list[0][0]
Pic2 = image_list[0][1]
### Bitwise
bitwise_addition = cv.bitwise_and(Pic1, Pic2)
### Print
cv.imshow("Bitwise_Addition",bitwise_addition)
cv.waitKey()
cv.destroyAllWindows() 


## Line
import format_iq_imgs as fii
import numpy as np
import cv
from pathlib import Path
import os

os.getcwd()
IMG_DIR = Path("../../example-data/iq-test/dmi-api-test")
IMG_DIR.exists()

img_files = list(IMG_DIR.glob("*image*.png"))
test_img = fii.read_img(img_files[5])

image_list = fii.split_img(test_img)

#Addition
Pic1 = image_list[0][0]
Pic2 = image_list[0][1]
### Bitwise
bitwise_addition = cv.bitwise_and(Pic1, Pic2)
### Greyscale
bitwise_addition = cv.cvtColor(bitwise_addition, cv.COLOR_RGBA2GRAY)
### Print
cv.imshow("Bitwise_Addition",bitwise_addition)
cv.waitKey()
cv.destroyAllWindows() 

# Soften
import format_iq_imgs as fii
import numpy as np
import cv as cv
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


# Color dots
# import the necessary packages
import format_iq_imgs as fii
import numpy as np
import cv as cv
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
	([0, 100, 100], [60, 60, 200])
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






# Stars
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

import matplotlib.pyplot as plt

os.getcwd()
IMG_DIR = Path("../../example-data/iq-test/dmi-api-test")
IMG_DIR.exists()

img_files = list(IMG_DIR.glob("*image*.png"))
test_img = fii.read_img(img_files[0])
image_list = fii.split_img(test_img)
# Image to alter
image = image_list[0][0]

# Convert BGR to HSV
#hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# https://answers.opencv.org/question/229620/drawing-a-rectangle-around-the-red-color-region/

# red color boundaries [B, G, R]
lower = [0, 0, 50]
upper = [30, 30, 200]

lower = np.array(lower, dtype="uint8")
upper = np.array(upper, dtype="uint8")

mask = cv.inRange(image, lower, upper)
output = cv.bitwise_and(image, image, mask=mask)

kernel = np.ones((5,5),np.uint8)
output = cv.dilate(output, kernel, iterations=3)

# Rotate Triangle
# grab the dimensions of the image and calculate the center of the
# image
(h, w) = output.shape[:2]
(cX, cY) = (w // 2, h // 2)
# rotate our output image by 144 degrees around the center of the image
M = cv.getRotationMatrix2D((cX, cY), 144, 1.0)
rotated = cv.warpAffine(output, M, (w, h))
cv.imshow("Rotated by 144 Degrees", rotated)
cv.waitKey(0)
