from asyncore import close_all
from multiprocessing.connection import wait
from pydoc import describe
import cv2
import numpy as np

# HSV colors
hsv_colors = np.array([[0, 0, 0], [0, 0, 85], [0, 0, 170], [0, 0, 255],
                       [0, 85, 0], [0, 85, 85], [0, 85, 170], [0, 85, 255],
                       [0, 170, 0], [0, 170, 85], [0, 170, 170], [0, 170, 255],
                       [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
                       [85, 0, 0], [85, 0, 85], [85, 0, 170], [85, 0, 255],
                       [85, 85, 0], [85, 85, 85], [85, 85, 170], [85, 85, 255],
                       [85, 170, 0], [85, 170, 85], [85, 170, 170],
                       [85, 170, 255], [85, 255, 0], [85, 255, 85],
                       [85, 255, 170], [85, 255, 255],
                       [170, 0, 0], [170, 0, 85], [170, 0, 170], [170, 0, 255],
                       [170, 85, 0], [170, 85, 85], [170, 85, 170],
                       [170, 85, 255], [170, 170, 0], [170, 170, 85],
                       [170, 170, 170], [170, 170, 255], [170, 255, 0],
                       [170, 255, 85], [170, 255, 170], [170, 255, 255],
                       [255, 0, 0], [255, 0, 85], [255, 0, 170], [255, 0, 255],
                       [255, 85, 0], [255, 85, 85], [255, 85, 170],
                       [255, 85, 255], [255, 170, 0], [255, 170, 85],
                       [255, 170, 170], [255, 170, 255]])


def check_matrix(img_list, choices):
    #test_list = img_list[:3]
    #final_imgs = img_list[3][:2]
    for row in img_list:
        for img in row:
            # find contours
            find_shapes_in_image(img, False)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def find_shapes_in_image(img, debug=False):
    used_colors = set()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, threshed = cv2.threshold(img_gray, 100, 255,
                                 cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    if debug:
        cv2.imshow("threshed", threshed)
        cv2.waitKey(1000)

    # get image dimensions
    height, width = threshed.shape
    rowsize = round(height / 6)
    colsize = round(width / 6)

    rows = np.array(range(rowsize, height - round(rowsize / 2), rowsize))
    cols = np.array(range(colsize, width - round(rowsize / 2), colsize))

    cnts = cv2.findContours(threshed, cv2.RETR_LIST,
                            cv2.CHAIN_APPROX_SIMPLE)[-2]
    matrix = np.zeros((5, 5))
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    # sort by row
    (cnts, boundingBoxes) = zip(
        *sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][0]))
    # sort by column
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(
        *sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][1]))
    for cnt in cnts:
        M = cv2.moments(cnt)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        row = find_nearest(rows, cY)
        col = find_nearest(cols, cX)
        # add to matrix if area is big enough
        if cv2.contourArea(cnt) > 20:
            # get contents of contour from img
            x, y, w, h = cv2.boundingRect(cnt)
            cnt_img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0),
                                    2)
            cnt_hsv = cv2.cvtColor(cnt_img, cv2.COLOR_BGR2HSV)
            cnt_hsv_avg = np.average(cnt_hsv, axis=0).astype(np.uint8)
            cnt_hsv_avg = np.average(cnt_hsv_avg, axis=0).astype(np.uint8)
            cnt_hsv_avg = np.average(cnt_hsv_avg, axis=0).astype(np.uint8)
            # get index of nearest color from hsv_colors
            cnt_index = find_nearest(hsv_colors, cnt_hsv_avg)
            used_colors.add(cnt_index)
            matrix[row][col] = len(used_colors)
    return matrix
