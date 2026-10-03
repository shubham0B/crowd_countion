import os
import sys
import shutil
import tempfile
import cv2
import numpy as np

# Ensure utf-8 stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.studio_tab import render_boxes_on_bgr, save_image_and_labels_to_disk, load_existing_boxes_from_dataset

def test_studio_workflow():
    test_dir = tempfile.mkdtemp(prefix="test_studio_")
    try:
        # 1. Create a dummy test image
        img_path = os.path.join(test_dir, "test_student_crowd.jpg")
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(dummy_img, "Test Crowd", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imwrite(img_path, dummy_img)

        # 2. Test rendering boxes
        test_boxes = [(50, 50, 150, 200), (200, 100, 300, 250)]
        vis = render_boxes_on_bgr(dummy_img, test_boxes)
        assert vis is not None, "Rendered image is None"
        assert vis.shape == dummy_img.shape, "Rendered image shape mismatch"
        print("[SUCCESS] render_boxes_on_bgr passed!")

        # 3. Test saving to training dataset
        dataset_output_dir = os.path.join(test_dir, "test_dataset")
        save_image_and_labels_to_disk(img_path, test_boxes, dataset_output_dir, is_val=False)

        # Verify image was saved
        saved_img_p = os.path.join(dataset_output_dir, "images", "train", "test_student_crowd.jpg")
        assert os.path.exists(saved_img_p), f"Saved image not found at {saved_img_p}"

        # Verify label was saved
        saved_lbl_p = os.path.join(dataset_output_dir, "labels", "train", "test_student_crowd.txt")
        assert os.path.exists(saved_lbl_p), f"Saved label not found at {saved_lbl_p}"

        with open(saved_lbl_p, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 2, f"Expected 2 lines in label file, got {len(lines)}"
            for l in lines:
                parts = l.strip().split()
                assert len(parts) == 5, f"Expected 5 parts in YOLO format line, got {parts}"
                assert parts[0] == "0", "Class ID should be 0"

        # Verify preview was saved
        saved_prev_p = os.path.join(dataset_output_dir, "previews", "prev_test_student_crowd.jpg")
        assert os.path.exists(saved_prev_p), f"Saved preview not found at {saved_prev_p}"

        # Verify college_data.yaml was generated
        yaml_p = os.path.join(dataset_output_dir, "college_data.yaml")
        assert os.path.exists(yaml_p), f"college_data.yaml not found at {yaml_p}"

        # 4. Test loading existing boxes from saved dataset
        loaded_boxes = load_existing_boxes_from_dataset(saved_img_p, dataset_output_dir)
        assert len(loaded_boxes) == 2, f"Expected 2 loaded boxes, got {len(loaded_boxes)}"
        print("[SUCCESS] save_image_and_labels_to_disk & load_existing_boxes_from_dataset passed!")

        print("[SUCCESS] ALL STUDIO WORKFLOW TESTS PASSED PERFECTLY!")
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)

if __name__ == "__main__":
    test_studio_workflow()
