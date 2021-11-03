from typing import List, Tuple
from pathlib import Path
from PIL import Image
import cv2
import torch
import itertools
import generate_imgs
import numpy as np

TEST_PATH = Path("../example-data/wheres-waldo/waldo.jpg")
MODEL = torch.hub.load("ultralytics/yolov5", "custom", path="waldominator_v3.pt")


def show_img(img):
    cv2.imshow("shown", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def PIL_to_cv2(pil_image):
    """ Converts PIL image to cv2 format (BGR) """
    cv2_img = np.array(pil_image)
    return cv2_img[:, :, ::-1].copy()


def crop_img(img: np.ndarray, pos: Tuple[int], img_size=(300, 300)) -> np.ndarray:
    x = pos[1] * img_size[1]
    y = pos[0] * img_size[0]
    return img[y : y + img_size[0], x : x + img_size[1]]


def split_img_matrix(input_img: Image, img_size=(300, 300)) -> List[np.ndarray]:
    img = np.array(input_img)
    width = int(img.shape[1] / img_size[1])
    height = int(img.shape[0] / img_size[0])
    positions = list(itertools.product(range(height), range(width)))
    return [crop_img(img, pos) for pos in positions], positions


def get_center_coords(bbox):
    x_mid = (bbox[0] + bbox[2]) / 2
    y_mid = (bbox[1] + bbox[3]) / 2
    return (x_mid, y_mid)


def tensor_to_xy(bbox: torch.tensor, position):
    newbox = bbox.numpy()[0, :4]
    x_mid, y_mid = get_center_coords(newbox)
    x = position[1] * 300 + x_mid
    y = position[0] * 300 + y_mid
    return x, y


def find_tensor_pos(preds):
    confidence = [float(pred[0][4]) if pred.nelement() > 0 else 0 for pred in preds]
    return np.argmax(confidence)


def eval_pred(x, y, full_bbox):
    x_good = full_bbox[0] < x < full_bbox[2]
    y_good = full_bbox[1] < y < full_bbox[3]
    return x_good and y_good


def predict(img: np.ndarray) -> Tuple[float, float]:
    img_list, pos = split_img_matrix(img)
    output = MODEL(img_list)
    idx = find_tensor_pos(output.pred)
    pos_final = pos[idx]
    try:
        return tensor_to_xy(output.pred[idx], pos_final)
    except IndexError:
        return 750.0, 750.0


def plot_circle(x, y, img):
    return cv2.circle(img.copy(), (int(x), int(y)), radius=5, color=(0, 255, 0))


# Random shit, don't mind
if __name__ == "__main__":
    test_img = cv2.imread(str(TEST_PATH))
    show_img(test_img)

    test_image = Image.open(TEST_PATH).convert("RGB")
    PIL_to_cv2
    x, y = predict(test_image)

    cv2.imwrite("test_img.png", plot_circle(x, y, test_img))

    new_test = cv2.imread(
        str("C:\\Users\\jhr\\datasets1\\waldo\\images\\train\\00DUOAF6.jpg")
    )
    new_testy = new_test[:, :, ::-1]
    model_output = MODEL(new_testy, augment=True)
    model_output.pred
    model_output.show()
    show_img(new_test)

    annotations = list(generate_imgs.ANN_DIR.glob("*.xml"))
    ann_dict = generate_imgs.create_ann_dict(annotations)

    test_img2, bbox = generate_imgs.create_training_img(ann_dict)
    img_list2, pos2 = split_img_matrix(test_img2)
    output = MODEL(img_list2)
    idx = find_tensor_pos(output.pred)
    pos = pos2[idx]
    x, y = tensor_to_xy(output.pred[idx], pos)
    eval_pred(x, y, bbox)

    output.pred[idx]
    pos

    score = 0
    N = 10
    for i in range(N):
        test_img2, bbox = generate_imgs.create_training_img(ann_dict)
        img_list2, pos2 = split_img_matrix(test_img2)
        output = MODEL(img_list2)
        idx = find_tensor_pos(output.pred)
        pos = pos2[idx]
        try:
            x, y = tensor_to_xy(output.pred[idx], pos)
        except IndexError:
            score += 0
            continue
        score += eval_pred(x, y, bbox)
