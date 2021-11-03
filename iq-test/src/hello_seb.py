import format_iq_imgs as fii
import cv2

img_files = list(fii.IMG_DIR.glob("*image*.png"))
test_img = fii.read_img(img_files[0])

image_list = fii.split_img(test_img)

fii.show_img(image_list[0][0])

print(image_list[0][0].shape)
# (225, 400, 3)
cv2.rotate(image_list[0][0])
