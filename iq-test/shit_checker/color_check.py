import shit_checker.format_iq_imgs as fii
import shit_checker.rotate_check as rc
import cv2
import numpy as np
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans

from skimage.metrics import structural_similarity as compare_ssim


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


if __name__ == "__main__":
    img_paths = rc.find_img_files()
    for i, img_path in enumerate(img_paths):
        choice_paths = rc.find_img_choices(img_path)
        choices = [fii.read_img(f) for f in choice_paths]
        img = fii.read_img(img_path)
        img_list = fii.split_img(img)
        print(check_bitand(img_list, choices))

    np.logical_xor(np.array([1, 1, 0, 1]), np.array([0, 1, 0, 1]))

    fii.show_img(img)
    choice = choices[0]

    source1 = img_list[0][0]
    source2 = img_list[0][1]
    bitstuff = xor_func(source1, source2)
    [compare_ssim(bitstuff, choice, multichannel=True) for choice in choices]
    fii.show_img(xor_preprocess(choice))
    fii.show_img(bitstuff)
    choice

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

