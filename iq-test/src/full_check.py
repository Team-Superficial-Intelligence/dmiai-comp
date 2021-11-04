import random

import format_iq_imgs as fii
import color_check as cc
import rotate_check as rc
import red_dot_check as rd
import rounding_check as ro


def check_shit(img_path):
    choice_paths = rc.find_img_choices(img_path)
    choices = [fii.read_img(f) for f in choice_paths]
    img = fii.read_img(img_path)
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


img_paths = rc.find_img_files()
for img_path in img_paths:
    print(check_shit(img_path))
