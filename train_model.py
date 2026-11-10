# train_model.py
from roboflow import Roboflow
from ultralytics import YOLO
import torch

def main():
    # Check CUDA availability
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        print("⚠️ GPU not detected, training will use CPU.")

    # 1. Download dataset from Roboflow (only first time)
    rf = Roboflow(api_key="vJQzLlTg3NyBBu9Z8JEw")
    project = rf.workspace("divya-lzcld").project("taco-mqclx")
    version = project.version(3)
    dataset = version.download("yolov9")   # Saves folder as "TACO-3/"

    # 2. Train YOLOv10 model
    model = YOLO("yolov10n.pt")
    results = model.train(
        data="TACO-3/data.yaml",
        epochs=10,
        imgsz=640,
        batch=32,
        plots=True,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    print("✅ Training complete! Best weights saved in runs/detect/train/weights/best.pt")

if __name__ == "__main__":
    main()
