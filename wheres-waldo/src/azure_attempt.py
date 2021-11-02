from azure.cognitiveservices.vision.customvision.training import (
    CustomVisionTrainingClient,
)
from azure.cognitiveservices.vision.customvision.prediction import (
    CustomVisionPredictionClient,
)
from azure.cognitiveservices.vision.customvision.training.models import (
    ImageFileCreateBatch,
    ImageFileCreateEntry,
    Region,
)
from msrest.authentication import ApiKeyCredentials
import os, time, uuid
import json
import cv2
from pathlib import Path
from typing import List


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def reformat_bbox(bbox: List[int], img_shape) -> List[float]:
    """ 
    Inputs a bounding box with coord (xmin, ymin, xmax, ymax)
    transforms into microsoft format (normalized left, top, width, height)
    """
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return [
        bbox[0] / img_shape[1],
        bbox[1] / img_shape[0],
        width / img_shape[1],
        height / img_shape[0],
    ]


configs = load_json(Path("../../../azure_keys.json"))

ENDPOINT = configs["endpoint"]
training_key = configs["train_key"]
TRAINING_DIR = Path("../../example-data/wheres-waldo/training_imgs")

# Auth
credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
trainer = CustomVisionTrainingClient(ENDPOINT, credentials)

# Create project
publish_iteration_name = "waldominator1"

# Find the object detection domain
obj_detection_domain = next(
    domain
    for domain in trainer.get_domains()
    if domain.type == "ObjectDetection" and domain.name == "General"
)

# Create a new project
print("Creating project...")
# Use uuid to avoid project name collisions.
project = trainer.create_project(str(uuid.uuid4()), domain_id=obj_detection_domain.id)

# add tags
waldo_tag = trainer.create_tag(project.id, "waldo")

# Loading data
bbox_paths = list((TRAINING_DIR / "bboxes").glob("*.json"))
bboxes = [load_json(bbox_path) for bbox_path in bbox_paths]

img_paths = list(TRAINING_DIR.glob("*.png"))
img_shape = cv2.imread(str(img_paths[0])).shape

waldo_image_regions = {
    filename: reformat_bbox(bbox, img_shape)
    for bbox_dict in bboxes
    for filename, bbox in bbox_dict.items()
}
