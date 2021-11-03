import format_iq_imgs as fii
import numpy as np
import cv2
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
bitwise_addition = cv2.bitwise_and(Pic1, Pic2)
### Print
cv2.imshow("Bitwise_Addition",bitwise_addition)
cv2.waitKey()
cv2.destroyAllWindows() 

### Erode
kernel = np.ones((5,5),np.uint8)
Pic1 = cv2.erode(Pic1,kernel,iterations=3)
Pic2 = cv2.erode(Pic2,kernel,iterations=3)
### BLUR
Pi1 = cv2.GaussianBlur(Pic1,(7,7),cv2.BORDER_DEFAULT)
Pic2 = cv2.GaussianBlur(Pic2,(7,7),cv2.BORDER_DEFAULT)
### Greyscale
Pic1 = cv2.cvtColor(Pic1, cv2.COLOR_RGBA2GRAY)
Pic2 = cv2.cvtColor(Pic2, cv2.COLOR_RGBA2GRAY)

# Show One picture
SigleImageLookup = image_list[0][1]
cv2.imshow("Image",SigleImageLookup)
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

## IDEAS
### Greyscale
cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
### Dialate features after having "Eroded" and erased other
kernel = np.ones((5,5),np.uint8)
eroded = cv2.erode(dialated,kernel,iterations=1)
dialated = cv2.dilate(image,kernel,iterations=5)

### Edge cascade
blur = cv2.GaussianBlur(img,(7,7),cv.BORDER_DEFAULT)
canny = cv.Canny(blur,125,175

## Softness
circle = cv.circle(blank.copy(),(200,200),200,255,-1)
small_circlecv.circle(blank.copy(),(200,200),100,255,-1)
Triangle = cv.triangle(blank.copy(),(200,200),200,255,-1)
Triangle = cv.rectangle(blank.copy(),(30,30),(370,370),255,-1)
## Combine to soften
bitwise_and = cv2.bitwise_and(img, circle)
