import unittest
import numpy as np
import cv2
import json
from roug_ml.annotations.labelme_utl import create_mask, parse_labelme_json


class TestFunctions(unittest.TestCase):

    def test_create_mask(self):
        shapes = [{"points": [[10, 10], [50, 10], [50, 50], [10, 50]]}]
        image_shape = (100, 100, 3)
        expected_mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.fillPoly(
            expected_mask,
            [np.array([[10, 10], [50, 10], [50, 50], [10, 50]], dtype=np.int32)],
            color=255,
        )

        mask = create_mask(shapes, image_shape)

        self.assertTrue(np.array_equal(mask, expected_mask))

    def test_parse_labelme_json(self):
        json_data = {
            "shapes": [{"points": [[10, 10], [50, 10], [50, 50], [10, 50]]}],
            "imagePath": "image.jpg",
        }
        with open("test.json", "w") as f:
            json.dump(json_data, f)

        shapes, image_path = parse_labelme_json("test.json")

        self.assertEqual(shapes, json_data["shapes"])
        self.assertEqual(image_path, json_data["imagePath"])

        # Clean up the temporary test file
        import os

        os.remove("test.json")


if __name__ == "__main__":
    unittest.main()
