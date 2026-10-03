#!/usr/bin/env python3
"""
train_college_model.py
=============================================================================
Fine-tunes YOLOv8 on your auto-labeled College Crowd dataset.
=============================================================================
Usage:
  python train_college_model.py --data college_dataset/college_data.yaml --epochs 40
=============================================================================
"""

import os
import sys
import argparse
from ultralytics import YOLO
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from config import DEVICE, MODELS_DIR

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on College Crowd Data")
    parser.add_argument("--data", type=str, default="college_dataset/college_data.yaml",
                        help="Path to college_data.yaml")
    parser.add_argument("--model", type=str, default="yolov8l.pt",
                        help="Base YOLO model weights (yolov8l.pt or yolov8m.pt)")
    parser.add_argument("--epochs", type=int, default=40,
                        help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=1280,
                        help="Input resolution for training (1280 recommended for crowds)")
    parser.add_argument("--batch", type=int, default=4,
                        help="Batch size (reduce to 2 or 4 if CPU or out of memory)")
    parser.add_argument("--name", type=str, default="college_crowd_model",
                        help="Experiment name")
    args = parser.parse_args()

    yaml_path = os.path.abspath(args.data)
    if not os.path.exists(yaml_path):
        print(f"❌ Error: Dataset config '{yaml_path}' not found!")
        print("👉 First run: python auto_label_college_data.py --input <your_data_folder>")
        return

    print("=" * 60)
    print("🎓 TRAINING CUSTOM YOLO MODEL ON COLLEGE CROWD DATA")
    print("=" * 60)
    print(f"📄 Dataset YAML:    {yaml_path}")
    print(f"🧠 Base Model:      {args.model}")
    print(f"🔄 Epochs:          {args.epochs}")
    print(f"📐 Image Size:      {args.imgsz}")
    print(f"📦 Batch Size:      {args.batch}")
    print(f"⚡ Device:          {DEVICE.upper()}")
    print("-" * 60)

    # Initialize model
    model = YOLO(args.model)

    # Start training
    results = model.train(
        data=yaml_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=DEVICE,
        name=args.name,
        save=True,
        workers=2,
        augment=True,
        mosaic=1.0,        # Mosaic data augmentation (great for small crowd people)
        mixup=0.1,
        degrees=10.0,
        flipud=0.0,
        fliplr=0.5
    )

    best_weight = os.path.join("runs", "detect", args.name, "weights", "best.pt")
    target_weight = os.path.join(MODELS_DIR, "yolov8l_college_custom.pt")

    if os.path.exists(best_weight):
        import shutil
        shutil.copy2(best_weight, target_weight)
        print("\n" + "=" * 60)
        print("🎉 TRAINING COMPLETED!")
        print(f"🏆 Best weights saved to: {target_weight}")
        print("=" * 60)
        print("👉 Your new custom model is ready to use in the Streamlit UI!")
    else:
        print(f"\nTraining completed. Weights are in runs/detect/{args.name}/weights/")

if __name__ == "__main__":
    main()
