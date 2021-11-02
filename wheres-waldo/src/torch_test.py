import torch
import generate_imgs as gi
import cv2
from pathlib import Path


model_path = Path("../../../yolov5/runs/train/exp4/weights/best.torchscript.pt")
data_path = Path("../../../datasets/waldo/images/val")

model = torch.jit.load(str(model_path))
test_img = next(data_path.glob("*.jpg"))

test_result = model(str(test_img))

gi.l
