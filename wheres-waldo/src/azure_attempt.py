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
from pathlib import Path


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


configs = load_json(Path("../../../azure_keys.json"))

ENDPOINT = configs["endpoint"]
training_key = configs["train_key"]

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
