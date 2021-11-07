from pathlib import Path
import shit_checker.full_check as fc
import shit_checker.rotate_check as rc

img_dir = Path("../example-data/iq-test/dmi-api-test")
img_paths = rc.find_img_files(img_path=img_dir)
for img_path in img_paths:
    img = fc.read_img_string(img_path)
    choice_paths = rc.find_img_choices(img_path, img_dir=img_dir)
    choices = [fc.read_img_string(img_file) for img_file in choice_paths]
    print(fc.check_shit(img, choices))
