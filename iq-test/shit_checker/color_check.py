from numpy.random.mtrand import noncentral_chisquare
import shit_checker.format_iq_imgs as fii
import shit_checker.rotate_check as rc
import cv2
import numpy as np
import imutils
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans

from skimage.metrics import structural_similarity as compare_ssim


# green color boundaries [B, G, R]
green = (np.array([20, 100, 40]), np.array([120, 255, 140]))
# Yellow color boundaries [B, G, R]
yellow = (np.array([0, 100, 155]), np.array([100, 150, 255]))
# Blue color boundaries [B, G, R]
blue = (np.array([120, 120, 30]), np.array([230, 200, 205]))
red = (np.array([0, 0, 50]), np.array([30, 30, 200]))
black = (np.array([0, 0, 0]), np.array([50, 50, 50]))
white = (np.array([210, 210, 210]), np.array([255, 255, 255]))

BOUNDARIES = [green, yellow, blue, red, black, white]


def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def apply_contrast(input_img, contrast=0):
    buf = input_img.copy()
    f = 131 * (contrast + 127) / (127 * (131 - contrast))
    alpha_c = f
    gamma_c = 127 * (1 - f)
    buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)
    return buf


def xor_preprocess(img):
    return to_gray(apply_contrast(img, contrast=130))


def xor_func(img1, img2):
    new_img1 = funky_func(img1)
    new_img2 = funky_func(img2)
    return cv2.bitwise_xor(new_img1, new_img2)


def bitand(src1, src2):
    return cv2.bitwise_and(src1, src2)


def bitand_list(img_list):
    return [
        compare_ssim(bitand(lst[0], lst[1]), lst[2], multichannel=True)
        for lst in img_list
    ]


def check_bitand(img_list, choices):
    test_list = img_list[:3]
    final_imgs = img_list[3][:2]
    crit = np.mean(bitand_list(test_list)) > 0.83
    if crit:
        best_guess = bitand(final_imgs[0], final_imgs[1])
        return np.argmax(compare_ssim(best_guess, choice) for choice in choices)
    return None


def check_xor(full_list, choices):
    test_case = full_list[0]
    final_imgs = full_list[3][:2]
    score = compare_ssim(xor_func(test_case[0], test_case[1]), funky_func(test_case[2]))
    if score > 0.95:
        best_guess = xor_func(final_imgs[0], final_imgs[1])
        return np.argmax(
            [compare_ssim(best_guess, funky_func(choice)) for choice in choices]
        )
    return None


def quantize_image(image, clusters=4):
    (h, w) = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # reshape the image into a feature vector so that k-means
    # can be applied
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    # apply k-means using the specified number of clusters and
    # then create the quantized image based on the predictions
    clt = MiniBatchKMeans(n_clusters=clusters)
    labels = clt.fit_predict(image)
    quant = clt.cluster_centers_.astype("uint8")[labels]
    # reshape the feature vectors to images
    quant = quant.reshape((h, w, 3))
    # convert from L*a*b* to RGB
    return cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)


def funky_func(src):
    test_img = quantize_image(src)
    unique_vals = np.unique(
        test_img.reshape(-1, test_img.shape[-1]), axis=0, return_counts=True
    )
    val1 = unique_vals[0][1]
    test_img[np.where((test_img != val1).all(axis=2))] = [0, 0, 0]
    test_img[np.where((test_img == val1).all(axis=2))] = [255, 255, 255]
    return to_gray(test_img)


def cnt_size(cnt) -> int:
    """ Returns the size of a contour """
    _, _, w, _ = cv2.boundingRect(cnt)
    return w


def calc_precedence(x, y, cols, tolerance_factor=10):
    return ((y // tolerance_factor) * tolerance_factor) * cols + x


def get_contour_precedence(contour, cols: int):
    """ Sorts the contours from top left to bottom right"""
    origin = cv2.boundingRect(contour)
    return calc_precedence(origin[0], origin[1], cols)


def get_cnts(img):
    """ Extracts nicely formatted contours from an image """
    imgray = to_gray(img)
    _, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(contours)
    cnts = [cnt for cnt in cnts if 5 < cnt_size(cnt) < 80]
    cnts.sort(key=lambda x: get_contour_precedence(x, img.shape[1]))
    return cnts


def count_color(img, bound):
    """ Checks how much of a color is in an img """
    return np.sum(cv2.inRange(img, bound[0], bound[1]))


def max_color(img):
    bnd_count = [count_color(img, bound) for bound in BOUNDARIES]
    return np.argmax(bnd_count)


def find_cnt_color(image, cnt):
    """ Finds the most prevalent color in a contour """
    x, y, w, h = cv2.boundingRect(cnt)
    img_crop = image[y : y + h, x : x + w]
    return max_color(img_crop)


def find_circle_color(img, circ):
    xmin = circ[0] - circ[2]
    xmax = circ[0] + circ[2]
    ymin = circ[1] - circ[2]
    ymax = circ[1] + circ[2]
    img_crop = img[ymin:ymax, xmin:xmax]
    return max_color(img_crop)


def cnt_to_nums(img, cnts):
    """ Converts contours to numbers based on color"""
    num_array = np.array([find_cnt_color(img, cnt) for cnt in cnts])
    return num_array


def correct_cols(array_list):
    """ if only two colours create a boolean """
    new_arrays = array_list
    unique_cols = np.unique(np.concatenate(array_list))
    if len(unique_cols) <= 2:
        new_arrays = [a == unique_cols[0] for a in array_list]
    return new_arrays


def final_logic_guess(final_imgs, choices, func):
    choice_arrs = correct_cols([convert_to_nums(choice) for choice in choices])
    final_arrs = correct_cols([convert_to_nums(img) for img in final_imgs])
    guess = func(*final_arrs)
    choice_truth = [(guess == arr).all() for arr in choice_arrs]
    return np.argmax(choice_truth) if np.any(choice_truth) else None


def check_logic_func(a_list, final_imgs, choices, func):
    result = [arr_check(*lst, func=func) for lst in a_list]
    if result:
        guess = final_logic_guess(final_imgs, choices, func)
        return guess
    return None


def not_xor(src1, src2):
    return ~np.logical_xor(src1, src2)


def not_and(src1, src2):
    return ~np.logical_and(src1, src2)


def color_logic_check(img_list, choices):
    test_list = img_list[:3]
    final_imgs = img_list[3][:2]
    # convert to nums
    a_list = [[convert_to_nums(img) for img in lst] for lst in test_list]
    a_list = [correct_cols(arrs) for arrs in a_list]
    func_list = [np.logical_xor, np.logical_and, not_xor, not_and]
    for func in func_list:
        result = check_logic_func(a_list, final_imgs, choices, func)
        if result is not None:
            return result
    return None


def convert_to_nums(img):
    cnts = get_cnts(img)
    return cnt_to_nums(img, cnts)


def arr_check(a1, a2, target, func=np.logical_xor):
    return np.all(func(a1, a2) == target)


def find_circles(img):
    gray = to_gray(img)
    minDist = 40
    param1 = 50  # 500
    param2 = 20  # 200 #smaller value-> more false circles
    minRadius = 4
    maxRadius = 20  # 10
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        1,
        minDist,
        param1=param1,
        param2=param2,
        minRadius=minRadius,
        maxRadius=maxRadius,
    )
    if circles is None:
        return None
    circles = list(np.round(circles[0, :]).astype("int"))
    circles.sort(key=lambda x: calc_precedence(x[0], x[1], img.shape[1]))
    return circles


if __name__ == "__main__":
    IMG_PATH = Path("../example-data/iq-test/dmi-api-test")
    img_paths = rc.find_img_files(img_path=IMG_PATH)
    img_path = img_paths[5]
    img = fii.read_img(img_path)
    img_list = fii.split_img(img)
    choice_paths = rc.find_img_choices(img_path, img_dir=IMG_PATH)
    choices = [fii.read_img(choice) for choice in choice_paths]
    test_img = img_list[0][0]
    fii.show_img(test_img)
    circles = find_circles(test_img)
    for (x, y, r) in circles:
        output = test_img.copy()
        circle_img = cv2.circle(output, (x, y), r, (0, 255, 0), 4)
        fii.show_img(output)
    cnts = get_cnts(test_img)
    for cnt in cnts:
        new_img = cv2.drawContours(test_img.copy(), [cnt], -1, (0, 255, 0), 2)
        fii.show_img(new_img)

    test_img = img_list[1][0]
    blurred = cv2.medianBlur(
        to_gray(test_img), 25
    )  # cv2.bilateralFilter(gray,10,50,50)
    fii.show_img(blurred)

    minDist = 40
    param1 = 50  # 500
    param2 = 1  # 200 #smaller value-> more false circles
    minRadius = 4
    maxRadius = 20  # 10
    fii.show_img
    # docstring of HoughCircles: HoughCircles(image, method, dp, minDist[, circles[, param1[, param2[, minRadius[, maxRadius]]]]]) -> circles
    circles = cv2.HoughCircles(
        to_gray(test_img),
        cv2.HOUGH_GRADIENT,
        1,
        minDist,
        param1=param1,
        param2=param2,
        minRadius=minRadius,
        maxRadius=maxRadius,
    )

    output = test_img.copy()
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = list(np.round(circles[0, :]).astype("int"))
        circles.sort(key=lambda x: calc_precedence(x[0], x[1], output.shape[1]))
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            circle_img = cv2.circle(output, (x, y), r, (0, 255, 0), 4)
            fii.show_img(output)
            # cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

    [find_circle_color(test_img, circle) for circle in circles]

    fii.show_img(np.hstack(output))
    test_row = img_list[0]
    test_arrs = [convert_to_nums(img) for img in test_row]

    source1 = img_list[0][0]
    source2 = img_list[0][1]
    bitstuff = xor_func(source1, source2)
    [compare_ssim(bitstuff, choice, multichannel=True) for choice in choices]
    fii.show_img(xor_preprocess(choice))
    fii.show_img(bitstuff)

    fii.show_img(quantize_image(source1, clusters=2))

    src1 = funky_func(source1)
    src2 = funky_func(source2)
    bitstuff = cv2.bitwise_xor(src1, src2)
    fii.show_img(bitstuff)

    testytest = funky_func(choice)
    fii.show_img(bitstuff)

    fii.show_img(testy)
    fii.show_img(test_img)

    deltaE
    hsvImg = cv2.cvtColor(choice, cv2.COLOR_BGR2HSV)

    # multiple by a factor to change the saturation
    hsvImg[..., 1] = hsvImg[..., 1] * 1.2

    image = cv2.cvtColor(hsvImg, cv2.COLOR_HSV2BGR)
    fii.show_img(image)

    fii.show_img(xor_func(source1, source1))
    cv2.COLORBGR2
    fii.show_img(bitstuff)

    kernel = np.ones((5, 5), np.uint8)
    fii.show_img(cv2.dilate(choices[1], kernel, iterations=1))

    for choice in choices:
        fii.show_img(choice)
    fii.show_img(bitstuff)
    target = img_list[0][2]

    gray = cv2.cvtColor(choice, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0.5)
    edge = cv2.Canny(blur, 0, 50, 3)

    contours, hierarchy = cv2.findContours(
        edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for contour, hier in zip(contours, hierarchy):
        (x, y, w, h) = cv2.boundingRect(contour)
        rect = choice[y : y + h, x : x + w]

        # rect = cv2.rectangle(choice, (x, y), (x + w, y + h), (0, 255, 0), 2)

    div = 4
    quantized = choice // div * div + div // 2

