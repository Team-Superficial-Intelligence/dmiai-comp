from asyncore import close_all
from math import ceil, floor, sqrt
from multiprocessing.connection import wait
from pydoc import describe
from typing import List, Tuple, Union
import cv2
import numpy as np

# HSV colors
hsv_colors = np.array([[190, 0, 10], [80, 160, 205], [190, 140, 80],
                       [80, 156, 58]])


def check_matrix(img_list, choices) -> None:
    matrices = get_matrix_representations(img_list, choices)
    return None
    if matrices is None:
        return None
    else:
        print("matrices found")
        return matrices


def get_matrix_representations(img_list, choices) -> Union[List, None]:
    # do some stuff
    test_cases = img_list[:3]
    final_imgs = img_list[3][:2]

    test_matrices = []
    final_matrices = []
    choice_matrices = []
    used_colors = []
    found_matrix = False
    for puzzle in test_cases:
        puzzles = []
        for img in puzzle:
            matrix, used_colors = find_shapes_in_image(img, used_colors, False)
            if matrix is not None:
                puzzles.append(matrix)
                found_matrix = True
        if len(puzzles) > 0:
            test_matrices.append(puzzles)
    for img in final_imgs:
        matrix, used_colors = find_shapes_in_image(img, used_colors, False)
        if matrix is not None:
            final_matrices.append(matrix)
            found_matrix = True
    for img in choices:
        matrix, used_colors = find_shapes_in_image(img, used_colors)
        if matrix is not None:
            choice_matrices.append(matrix)
            found_matrix = True

    if not found_matrix:
        return None
    return [test_matrices, final_matrices, choice_matrices]


def find_nearest(array, value):
    array = np.asarray(array)
    if type(value) == np.ndarray or type(value) == tuple:
        idx = (np.abs(array - value).sum(axis=1).argmin())
        return idx
    return (np.abs(array - value)).argmin()


def get_dominant_color(pixels, clusters, attempts):
    """
    Given a (N, Channels) array of pixel values, compute the dominant color via K-means
    """
    clusters = min(clusters, len(pixels))
    flags = cv2.KMEANS_RANDOM_CENTERS
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 1, 10)
    _, labels, centroids = cv2.kmeans(pixels.astype(np.float32), clusters,
                                      None, criteria, attempts, flags)
    _, counts = np.unique(labels, return_counts=True)
    dominant = centroids[np.argmax(counts)]
    return dominant


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0)**invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def find_shapes_in_image(img, used_colors=None, debug=False):
    if used_colors is None:
        used_colors = []

    # img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_gray = adjust_gamma(img_gray, 0.3)

    th, threshed = cv2.threshold(img_gray, 150, 255,
                                 cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # cv2.floodFill(threshed, None, (0, 0), 255)
    # cv2.floodFill(threshed, None, (0, 0), 0)

    cnts = cv2.findContours(threshed, cv2.RETR_LIST,
                            cv2.CHAIN_APPROX_SIMPLE)[-2]

    c = max(cnts, key=cv2.contourArea)
    # check for image border (must be greater than half the area)
    if cv2.contourArea(c) > (threshed.shape[0] * threshed.shape[1] / 2):
        mask = np.ones(threshed.shape[:2], dtype="uint8") * 255
        cv2.drawContours(mask, [c], -1, 0, 2)
        threshed = cv2.bitwise_and(threshed, threshed, mask=mask)

    #cv2.drawContours(threshed, cnts, -1, (0, 255, 0), 3)

    #cnts = cv2.findContours(threshed, cv2.RETR_LIST,
    #                        cv2.CHAIN_APPROX_SIMPLE)[-2]
    # if debug:
    #     # cv2.drawContours(threshed, cnts, -1, (0, 255, 0), 3)
    #     cv2.imshow('Contours', threshed)
    #     cv2.waitKey(1000)
    #grid_points = len(cnts)
    mindist = round(threshed.shape[0] / 8)
    maxr = round(threshed.shape[0] / 8)
    circles = cv2.HoughCircles(threshed,
                               cv2.HOUGH_GRADIENT,
                               1,
                               mindist,
                               param1=150,
                               param2=6.5,
                               minRadius=0,
                               maxRadius=maxr)

    if circles is None:
        return None, used_colors

    circles = np.uint16(np.around(circles))
    circles = circles[0, :]
    grid_points = len(circles)
    if grid_points < 9 or grid_points > 25:
        return None, used_colors

    row_count = int(sqrt(grid_points))
    col_count = row_count

    # get image dimensions
    height, width = threshed.shape
    rowsize = floor(height / (row_count + 1))
    colsize = floor(width / (col_count + 1))

    rows = np.array(range(rowsize, height - ceil(rowsize / row_count),
                          rowsize))
    cols = np.array(range(colsize, width - ceil(colsize / col_count), colsize))
    matrix = np.zeros((row_count, col_count))

    # boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    # sort by col, row
    circles = sorted(circles, key=lambda x: (x[1], x[0]))

    smallest_circle = min(circles, key=lambda x: x[2])
    smallest_circle_raidus = smallest_circle[2]

    largest_circle = max(circles, key=lambda x: x[2])
    largest_circle_radius = largest_circle[2]
    boundary = floor(smallest_circle_raidus +
                     (largest_circle_radius - smallest_circle_raidus) / 2)
    for circle in circles:

        # add to matrix if radius is large enough
        if circle[2] > boundary:
            col = find_nearest(cols, circle[0])
            row = find_nearest(rows, circle[1])
            # get contents of contour from img
            # only take radius /2, not full width
            w = ceil(circle[2] / 5)
            h = w
            x = circle[0] - 2
            y = circle[1] + ceil(circle[2] / 1.5)

            #cnt_img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0),
            #                        2)
            #cnt_img = cv2.circle(img, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
            cnt_img = img[y:y + h, x:x + w]
            if debug:
                cv2.imshow('Contours', cnt_img)
                cv2.waitKey(0)
            # cnt_hsv = cv2.cvtColor(cnt_img, cv2.COLOR_BGR2HSV)
            # cnt_img = adjust_gamma(cnt_img, 1.2)
            # cnt_hsv_avg = np.average(cnt_hsv, axis=0).astype(np.uint8)
            # cnt_hsv_avg = np.average(cnt_hsv_avg, axis=0).astype(np.uint8)
            # cnt_hsv_avg = np.average(cnt_hsv_avg, axis=0).astype(np.uint8)
            cnt_bgr_avg = cnt_img.mean(axis=0).mean(
                axis=0)  # cnt_hsv.mean(axis=0)
            cnt_rgb_avg = np.array([(cnt_bgr_avg[2], cnt_bgr_avg[1],
                                     cnt_bgr_avg[0])])
            # get index of nearest color from hsv_colors
            cnt_index = find_nearest(hsv_colors, cnt_rgb_avg)
            if cnt_index not in used_colors:
                used_colors.append(cnt_index)
            matrix[row][col] = used_colors.index(cnt_index) + 1
    return matrix, used_colors
