import numpy as np
import cv2
import rotate_check as rc
import format_iq_imgs as fii
from skimage.metrics import structural_similarity as compare_ssim


def show_img(img):
    cv2.imshow("s", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def make_circle(size=30):
    Circle = np.zeros([110, 110, 3], dtype=np.uint8)
    Circle.fill(255)  # numpy array!
    Circle = cv.circle(Circle, (55, 55), size, (0, 0, 0), -1)
    return Circle


def soften_func(img1, Circle):
    return cv.bitwise_or(img1, Circle)


def check_bitor(img_list, choices):
    Circle = make_circle(30)
    test_list = img_list[:3]
    final_imgs = img_list[3][:2]
    scores = [
        compare_ssim(soften_func(lst[0], Circle), lst[2], multichannel=True)
        for lst in test_list
    ]
    return scores
    # crit = np.mean(scores) > 0.7
    # if crit:
    #   best_guess = bitor(final_imgs[0], final_imgs[1])
    #  return np.argmax(compare_ssim(best_guess, choice) for choice in choices)
    # return None


rc.IMG_PATH

img_paths = rc.find_img_files()
img_path = img_paths[-2]
img = cv2.imread(str(img_path))
img_list = fii.split_img(img)
choices = rc.find_img_choices(img_path)

for i, img_path in enumerate(img_paths):
    img = cv2.imread(str(img_path))
    img_list = fii.split_img(img)
    choices = rc.find_img_choices(img_path)

    print(np.mean(check_bitor(img_list, choices)))
show_img(img)


############################ EXTRA STUFF #################################
# os.getcwd()
# IMG_DIR = Path("../../example-data/iq-test/dmi-api-test")
# IMG_DIR.exists()

# img_files = list(IMG_DIR.glob("*image*.png"))
# test_img = fii.read_img(img_files[3])
# image_list = fii.split_img(test_img)
## Circle
# Circle = np.zeros([110,110,3],dtype=np.uint8)
# Circle.fill(255) # numpy array!
# Circle= cv.circle(Circle,(55,55),30,(0,0,0),-1)
# Circle = Image.fromarray(Circle) #convert numpy array to image
# Image to alter
# TestImg = image_list[3][0]
# TestImg.shape
## Combine to "soften"
# Combined = cv.bitwise_or(TestImg,Circle)

