"""
This file contains function to deal with annotation coming from labelme.
To install label me uses pip install labelme on your venv
then run labelme by typing in terminal: labelme
"""

import json
import cv2
import numpy as np
from beartype.typing import List, Tuple


def create_mask(shapes: List[dict], image_shape: Tuple[int, int, int]) -> np.ndarray:
    """
    Create a binary mask from polygon annotations.

    :param shapes: A list of shape dictionaries, where each dictionary contains
                   the 'points' key with a list of coordinates (x, y) for the polygon vertices.
    :param image_shape: The shape of the image for which the mask is being created.
                         Expected format: (height, width, channels)

    :returns: A binary mask with polygons filled in. The mask is of the same height
                       and width as the input image, with a single channel.
    """
    mask = np.zeros(
        image_shape[:2], dtype=np.uint8
    )  # Assuming image_shape is (height, width, channels)

    for shape in shapes:
        points = np.array(shape["points"], dtype=np.int32)
        cv2.fillPoly(mask, [points], color=255)  # Fill the polygon

    return mask


def parse_labelme_json(json_file: str) -> Tuple[List, str]:
    """
    Parse a JSON file from LabelMe to extract annotation shapes and image path.

    :param json_file: The path to the LabelMe JSON file.

    :returns A tuple containing two elements:
           - shapes (list): A list of shape dictionaries from the LabelMe annotations.
           - image_path (str): The path to the annotated image.
    """
    with open(json_file, "r") as file:
        data = json.load(file)

    # Extract the annotation shapes
    shapes = data["shapes"]

    # Image file path
    image_path = data["imagePath"]
    return shapes, image_path
