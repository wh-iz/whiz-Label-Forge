import unittest
import numpy as np
import tempfile
import os
import cv2

from Main import (
    postprocess_predictions,
    rescale_boxes,
    resize_and_adjust_boxes,
    iou,
    merge_boxes,
    compute_phash,
    hamming_distance,
    is_similar_to_recent,
    save_labels,
)

class TestCoreAlgorithms(unittest.TestCase):
    def test_iou(self):
        box1 = [0, 0, 10, 10]
        box2 = [0, 0, 10, 10]
        self.assertAlmostEqual(iou(box1, box2), 1.0)

        box3 = [10, 10, 20, 20]
        self.assertAlmostEqual(iou(box1, box3), 0.0)

        box4 = [0, 0, 10, 5]
        # inter = 10 * 5 = 50, union = 100 + 50 - 50 = 100 -> iou = 0.5
        self.assertAlmostEqual(iou(box1, box4), 0.5)

    def test_weighted_box_fusion(self):
        # Two overlapping boxes for class 0 with different confidences
        b1 = [10.0, 10.0, 30.0, 30.0, 0.8, 0]
        b2 = [12.0, 12.0, 32.0, 32.0, 0.4, 0]
        merged = merge_boxes([b1, b2], merge_iou=0.4)
        self.assertEqual(len(merged), 1)
        # Expected weighted average: (10*0.8 + 12*0.4)/1.2 = (8.0 + 4.8)/1.2 = 12.8 / 1.2 ≈ 10.666
        self.assertAlmostEqual(merged[0][0], (10.0 * 0.8 + 12.0 * 0.4) / 1.2, places=3)
        self.assertAlmostEqual(merged[0][4], 0.8) # max conf
        self.assertEqual(merged[0][5], 0)

    def test_postprocess_yolov8_style(self):
        # YOLOv8 format: (1, 4 + C, N) -> e.g. (1, 6, 10) for 2 classes and 10 anchors
        # [cx, cy, w, h, cls0, cls1]
        preds = [np.zeros((1, 6, 10), dtype=np.float32)]
        # Anchor 0 has high score for class 1
        preds[0][0, 0, 0] = 100.0  # cx
        preds[0][0, 1, 0] = 100.0  # cy
        preds[0][0, 2, 0] = 50.0   # w
        preds[0][0, 3, 0] = 50.0   # h
        preds[0][0, 4, 0] = 0.1    # cls0
        preds[0][0, 5, 0] = 0.9    # cls1

        boxes = postprocess_predictions(preds, conf_thresh=0.5, iou_thresh=0.45)
        self.assertEqual(len(boxes), 1)
        b = boxes[0]
        self.assertAlmostEqual(b[0], 75.0)  # x1 = 100 - 25
        self.assertAlmostEqual(b[1], 75.0)  # y1 = 100 - 25
        self.assertAlmostEqual(b[2], 125.0) # x2 = 100 + 25
        self.assertAlmostEqual(b[3], 125.0) # y2 = 100 + 25
        self.assertAlmostEqual(b[4], 0.9)
        self.assertEqual(b[5], 1)

    def test_postprocess_yolov5_style(self):
        # YOLOv5 format: (1, N, 5 + C) -> e.g. (1, 10, 7) for 2 classes: [cx, cy, w, h, obj_conf, cls0, cls1]
        preds = [np.zeros((1, 10, 7), dtype=np.float32)]
        preds[0][0, 0] = [100.0, 100.0, 40.0, 40.0, 0.9, 0.1, 0.8]
        # Final conf for class 1: 0.9 * 0.8 = 0.72 >= 0.5
        boxes = postprocess_predictions(preds, conf_thresh=0.5, iou_thresh=0.45)
        self.assertEqual(len(boxes), 1)
        self.assertAlmostEqual(boxes[0][4], 0.72, places=3)
        self.assertEqual(boxes[0][5], 1)

    def test_postprocess_yolov10_style(self):
        # YOLOv10 format: (1, 300, 6) with [x1, y1, x2, y2, score, cls_id]
        preds = [np.zeros((1, 300, 6), dtype=np.float32)]
        preds[0][0, 0] = [10.0, 20.0, 50.0, 60.0, 0.85, 2.0]
        boxes = postprocess_predictions(preds, conf_thresh=0.5)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0][:4], [10.0, 20.0, 50.0, 60.0])
        self.assertAlmostEqual(boxes[0][4], 0.85)
        self.assertEqual(boxes[0][5], 2)

    def test_perceptual_hash_and_hamming(self):
        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        img2 = np.zeros((100, 100, 3), dtype=np.uint8)
        h1 = compute_phash(img1)
        h2 = compute_phash(img2)
        self.assertEqual(hamming_distance(h1, h2), 0)
        self.assertTrue(is_similar_to_recent(img1, [h2], similarity_threshold=5))

        # Different image
        img3 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        h3 = compute_phash(img3)
        self.assertIsInstance(hamming_distance(h1, h3), int)

    def test_center_crop_boxes(self):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        # Box centered at (960, 540)
        box = [910.0, 490.0, 1010.0, 590.0, 0.9, 0] # 100x100 box
        cropped, new_boxes = resize_and_adjust_boxes(frame, [box], (640, 640))
        self.assertEqual(cropped.shape, (640, 640, 3))
        self.assertEqual(len(new_boxes), 1)
        # Center in 640x640 is (320, 320), so box from 270 to 370
        self.assertAlmostEqual(new_boxes[0][0], 270.0)
        self.assertAlmostEqual(new_boxes[0][1], 270.0)
        self.assertAlmostEqual(new_boxes[0][2], 370.0)
        self.assertAlmostEqual(new_boxes[0][3], 370.0)

    def test_save_labels_empty_and_valid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_file = os.path.join(tmpdir, "test.txt")
            # Save empty
            save_labels([], 640, 640, save_file)
            self.assertTrue(os.path.exists(save_file))
            with open(save_file, "r") as f:
                self.assertEqual(f.read().strip(), "")

            # Save with box
            boxes = [[100.0, 100.0, 200.0, 200.0, 0.9, 0]]
            save_labels(boxes, 640, 640, save_file)
            with open(save_file, "r") as f:
                content = f.read().strip()
            self.assertTrue(content.startswith("0 "))

if __name__ == "__main__":
    unittest.main()
