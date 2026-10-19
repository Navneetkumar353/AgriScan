# train_yolo.py
from ultralytics import YOLO
from pathlib import Path
import json

DATA_ROOT = "data"  # expects data/train and data/val (or data/test)
IMGSZ = 320
EPOCHS = 15
BATCH = 32


def main():
    # Stronger backbone than 'n' for better top-1 at same epochs
    model = YOLO("yolov8s-cls.pt")

    # Train with regularization that helps when you can't change the dataset
    r = model.train(
        data=DATA_ROOT,
        imgsz=IMGSZ,
        epochs=EPOCHS,
        batch=BATCH,
        cos_lr=True,
        auto_augment="randaugment",
        mixup=0.1,
        cutmix=0.1,
        erasing=0.4,
        dropout=0.05,
        seed=0,
        verbose=True,
    )

    best = Path(r.save_dir) / "weights" / "best.pt"
    print("Best weights:", best)

    # Persist class order for inference
    names = model.model.names
    classes = [names[i] for i in range(len(names))]
    with open("class_names.json", "w", encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)
    print("Wrote class_names.json:", classes)

    # Quick validation to see exact top1/top5
    print("\n=== Validating best.pt ===")
    metrics = model.val(model=str(best), data=DATA_ROOT, imgsz=IMGSZ)
    print(f"top1={metrics.top1:.4f}  top5={metrics.top5:.4f}")


if __name__ == "__main__":
    main()
