import format_iq_imgs as fii
import color_check as cc
import rotate_check as rc

img_paths = rc.find_img_files()


def check_shit(img_path):
    choice_paths = rc.find_img_choices(img_path)
    choices = [fii.read_img(f) for f in choice_paths]
    img = fii.read_img(img_path)
    img_list = fii.split_img(img)

    rotation_result = rc.check_rotations(img_list, choices)
    if rotation_result:
        return rotation_result

    bitxor_result = cc.check_xor(img_list, choices)
    if bitxor_result:
        return bitxor_result

    bitand_result = cc.check_bitand(img_list, choices)
    if bitand_result:
        return bitand_result

    return None


for img_path in img_paths:
    img_path = img_paths[0]
    print(check_shit(img_path))
