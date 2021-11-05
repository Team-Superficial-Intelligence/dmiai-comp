"""
Checks whether the images are basically all the same. 
The strategy is then to select images that are basically similar also :)) 
(This probably works slightly better than random)
"""
import shit_checker.format_iq_imgs as fii
import shit_checker.rotate_check as rc
import itertools
import numpy as np
from pathlib import Path
from skimage.metrics import structural_similarity as compare_ssim


def all_img_list(img_path):
    img = fii.read_img(img_path)
    return fii.split_img(img)


def img_sim(img_list):
    return [compare_ssim(lst[1], lst[2], multichannel=True) for lst in img_list[:3]]


def choice_similarity(choices):
    choice_pairs = itertools.combinations(choices, 2)
    return [compare_ssim(*pair, multichannel=True) for pair in choice_pairs]


def check_semi_similar(full_list, choices):
    test_cases = full_list[:3]
    final_img = full_list[3][1]
    # Check if cases are pretty similar
    case_similarity = [choice_similarity(lst) for lst in test_cases]
    case_crit = 0.82 < np.mean(case_similarity) < 0.92
    # Check if choices are pretty dissimilar
    choice_sim = choice_similarity(choices)
    choice_crit = np.mean(choice_sim) < 0.8
    if choice_crit and case_crit:
        # Find the next most similar choice to final_img
        final_sim = [
            compare_ssim(final_img, choice, multichannel=True) for choice in choices
        ]
        return np.argsort(final_sim)[-2]
    return None


if __name__ == "__main__":
    img_dir = Path("../example-data/iq-test/dmi-api-test")
    image_paths = rc.find_img_files(img_path=img_dir)
    img_path = image_paths[-2]
    img = fii.read_img(img_path)
    img_list = fii.split_img(img)
    choices = rc.find_img_choices(img_path, img_dir=img_dir)
    choices = [fii.read_img(img) for img in choices]
    image_path = img_path
    for i, image_path in enumerate(image_paths):
        img_list = all_img_list(image_path)
        choice_paths = rc.find_img_choices(image_path, img_dir=img_dir)
        choices = [fii.read_img(img) for img in choice_paths]
        print(check_semi_similar(img_list, choices))
