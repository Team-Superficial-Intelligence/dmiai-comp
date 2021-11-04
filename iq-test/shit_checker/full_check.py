import base64
import random
import numpy as np
import cv2
from typing import List

import shit_checker.format_iq_imgs as fii
import shit_checker.color_check as cc
import shit_checker.rotate_check as rc

# import red_dot_check as rd
# import rounding_check as ro


def base64_to_cv2(img_string: str) -> np.array:

    nparr = np.fromstring(base64.b64decode(img_string), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def check_shit(img_string: str, choice_list: List[str]):
    img = base64_to_cv2(img_string)
    choices = [base64_to_cv2(choice) for choice in choice_list]
    img_list = fii.split_img(img)

    rotation_result = rc.check_rotations(img_list, choices)
    if rotation_result is not None:
        print("it's a rotation!")
        return rotation_result

    bitxor_result = cc.check_xor(img_list, choices)
    if bitxor_result is not None:
        print("colour funk going on!")
        return bitxor_result

    bitand_result = cc.check_bitand(img_list, choices)
    if bitand_result is not None:
        print("bitand galore!")
        return bitand_result
    print("let's go random!")
    return random.choice(range(len(choices)))


# img_paths = rc.find_img_files()
# for img_path in img_paths:
#    print(check_shit(img_path))
