import format_iq_imgs as fii
from pathlib import Path

IMG_PATH = Path("../../example-data/iq-test/dmi-api-test")

img_files = list(IMG_PATH.glob("*image*.png"))

img = fii.read_img(img_files[1])

