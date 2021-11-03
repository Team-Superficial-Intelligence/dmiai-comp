import torch
import generate_imgs
import cv2
from pathlib import Path


def read_txt(file_path: Path):
    with open(file_path, "r") as f:
        return f.readline()


def parse_text_box(text_box: str):
    return [float(x) for x in test_box.split()[1:]]


def yolo_to_bbox(yolo_bbox, img_dims=(300, 300)):
    x1 = (yolo_bbox[0] - yolo_bbox[2] / 2) * img_dims[0]
    y1 = (yolo_bbox[1] - yolo_bbox[3] / 2) * img_dims[1]
    x2 = (yolo_bbox[0] + yolo_bbox[2] / 2) * img_dims[0]
    y2 = (yolo_bbox[1] + yolo_bbox[3] / 2) * img_dims[1]
    return int(x1), int(y1), int(x2), int(y2)


def draw_yolo_bbox(img, yolo_bbox):
    x1, y1, x2, y2 = yolo_to_bbox(yolo_bbox)
    return cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)


model_path = Path("../../../yolov5/legend_best.pt")
data_path = Path("../../../datasets/waldo/images/train")
label_path = data_path.parent.parent / "labels" / "train"
model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path)


test_img = next(data_path.glob("*.jpg"))

results = model(list(data_path.glob("*.jpg"))[1])

results.pred
test_name = test_img.name[:-4]

test_box = next(label_path.glob(f"*{test_name}*"))


test_label = next(label_path.glob("*txt"))
test_box = read_txt(test_label)
parse_text_box(test_box)

test_img_path = next(data_path.glob(f"*{test_label.name[:-4]}.jpg"))
test_img = cv2.imread(str(test_img_path))

new_img = draw_yolo_bbox(test_img, parse_text_box(test_box))

generate_imgs.show_img(new_img)

yolo_to_bbox(parse_text_box(test_box))
