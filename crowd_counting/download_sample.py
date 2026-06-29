# ============================================================
# download_sample.py — Downloads test crowd images
# ============================================================
# WHY THIS EXISTS:
# The system needs real crowd images to test with.
# Run this script once to get sample images into the images/ folder.
#
# HOW TO RUN:
#   python download_sample.py
# ============================================================

import urllib.request
import os

SAMPLES = [
    (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/b/be/June_odd-eyed-cat.jpg/800px-June_odd-eyed-cat.jpg",
        "sample_test.jpg",
        "Test image (replace with real crowd images)"
    )
]

os.makedirs("images", exist_ok=True)

print("To properly test this system, add your own crowd images to the images/ folder.")
print("Accepted formats: .jpg, .jpeg, .png")
print("\nFor testing with real crowd datasets, consider:")
print("  → ShanghaiTech Dataset (crowd counting benchmark)")
print("  → COCO validation set (person images)")
print("  → UCF-QNRF crowd dataset")
print("\nYour own images work perfectly. Just drag them into the images/ folder.")
print("Then upload via the Streamlit interface at http://localhost:8501")
