"""
Checks whether we have the right dataset
"""
from pathlib import Path
import xml.etree.ElementTree as ET

ANN_PATH = Path("./extra_anns")
IMG_DIR = Path("../example-data/wheres-waldo/images")


def read_content(xml_file: str):
    """ Parses a XML file of PASCAL VOC Annotations"""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    list_with_all_boxes = []
    for boxes in root.iter("object"):
        filename = root.find("filename").text
        ymin, xmin, ymax, xmax = None, None, None, None
        ymin = int(boxes.find("bndbox/ymin").text)
        xmin = int(boxes.find("bndbox/xmin").text)
        ymax = int(boxes.find("bndbox/ymax").text)
        xmax = int(boxes.find("bndbox/xmax").text)
        list_with_single_boxes = [xmin, ymin, xmax, ymax]
        list_with_all_boxes.append(list_with_single_boxes)
    return {filename: list_with_all_boxes}


ann_list = [read_content(xml_file) for xml_file in ANN_PATH.glob("*.xml")]

