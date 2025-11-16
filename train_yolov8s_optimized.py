"""
YOLOv8s Training Script - Optimized for Small Objects Detection
Armeta CV Hackathon - Document Signature/Stamp/QR Detection

IMPROVEMENTS OVER YOLOv8n:
- YOLOv8s (11M params) instead of YOLOv8n (3M params)
- imgsz=1024 instead of 640 (critical for small QR codes)
- Copy-paste augmentation for small objects
- Lower learning rate for fine-tuning
- Class weights to handle imbalance
"""

from ultralytics import YOLO
import torch
import yaml
import os
from pathlib import Path

# ============= CONFIGURATION =============
CONFIG = {
    # Model
    'model': 'yolov8s.pt',  # ⭐ CHANGED from yolov8n.pt

    # Data
    'data_yaml': '/kaggle/working/data/data.yaml',  # Kaggle path (change if using Colab)

    # Training params
    'epochs': 120,           # ⬆️ Increased (more capacity)
    'imgsz': 1024,          # ⭐ CRITICAL: 640→1024 for small objects
    'batch': 4,             # ⬇️ Reduced (1024 uses more memory)
    'patience': 20,         # ⬆️ More patience for convergence
    'device': 0,            # GPU
    'amp': True,            # Mixed precision

    # Optimizer
    'optimizer': 'AdamW',   # Better than SGD for small datasets
    'lr0': 0.0001,         # ⬇️ Lower LR for fine-tuning (was 0.001)
    'lrf': 0.01,           # Final LR = lr0 * lrf
    'momentum': 0.937,
    'weight_decay': 0.0005,

    # Augmentation (optimized for small objects)
    'augment': True,
    'mosaic': 1.0,         # ⭐ Mosaic augmentation
    'copy_paste': 0.3,     # ⭐ Copy-paste small objects
    'mixup': 0.15,         # ⭐ Mixup augmentation
    'degrees': 5.0,        # ⬇️ Less rotation (preserve orientation)
    'translate': 0.1,      # ⬇️ Less translation (keep objects in frame)
    'scale': 0.9,          # ⬇️ Less scaling (preserve small objects)
    'fliplr': 0.5,         # Horizontal flip
    'flipud': 0.0,         # No vertical flip (documents)
    'hsv_h': 0.015,        # Hue augmentation
    'hsv_s': 0.7,          # Saturation
    'hsv_v': 0.4,          # Value

    # Loss weights
    'box': 7.5,            # BBox loss weight
    'cls': 0.5,            # Classification loss
    'dfl': 1.5,            # Distribution focal loss

    # Other
    'workers': 4,
    'save_period': 10,     # Save checkpoint every 10 epochs
    'verbose': True,
    'seed': 42,
    'deterministic': True,
}

# Class weights (to handle imbalance)
# Based on your data: signature=103, stamp=60, qr=95
# Weight = 1 / sqrt(count)
CLASS_WEIGHTS = {
    0: 1.0,   # signature (baseline)
    1: 1.3,   # stamp (fewer samples)
    2: 1.05,  # qr
}

# ============= MAIN =============
def main():
    print("=" * 60)
    print("YOLOv8s Training - Optimized for Small Objects")
    print("=" * 60)

    # Check GPU
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  No GPU detected! Training will be slow.")

    # Load model
    print(f"\n📦 Loading {CONFIG['model']}...")
    model = YOLO(CONFIG['model'])

    # Model info
    print(f"   Parameters: {sum(p.numel() for p in model.model.parameters()) / 1e6:.1f}M")

    # Check data.yaml exists
    if not os.path.exists(CONFIG['data_yaml']):
        raise FileNotFoundError(f"❌ data.yaml not found: {CONFIG['data_yaml']}")

    # Load data.yaml to check classes
    with open(CONFIG['data_yaml'], 'r') as f:
        data_config = yaml.safe_load(f)
    print(f"\n📊 Dataset:")
    print(f"   Classes: {data_config['names']}")
    print(f"   Train: {data_config['train']}")
    print(f"   Val: {data_config['val']}")

    # Start training
    print(f"\n🚀 Starting training with config:")
    print(f"   Model: {CONFIG['model']}")
    print(f"   Image size: {CONFIG['imgsz']}")
    print(f"   Batch size: {CONFIG['batch']}")
    print(f"   Epochs: {CONFIG['epochs']}")
    print(f"   Learning rate: {CONFIG['lr0']}")
    print(f"   Augmentation: mosaic={CONFIG['mosaic']}, copy_paste={CONFIG['copy_paste']}")
    print("-" * 60)

    # Train
    results = model.train(
        data=CONFIG['data_yaml'],
        epochs=CONFIG['epochs'],
        imgsz=CONFIG['imgsz'],
        batch=CONFIG['batch'],
        patience=CONFIG['patience'],
        device=CONFIG['device'],
        amp=CONFIG['amp'],
        optimizer=CONFIG['optimizer'],
        lr0=CONFIG['lr0'],
        lrf=CONFIG['lrf'],
        momentum=CONFIG['momentum'],
        weight_decay=CONFIG['weight_decay'],
        augment=CONFIG['augment'],
        mosaic=CONFIG['mosaic'],
        copy_paste=CONFIG['copy_paste'],
        mixup=CONFIG['mixup'],
        degrees=CONFIG['degrees'],
        translate=CONFIG['translate'],
        scale=CONFIG['scale'],
        fliplr=CONFIG['fliplr'],
        flipud=CONFIG['flipud'],
        hsv_h=CONFIG['hsv_h'],
        hsv_s=CONFIG['hsv_s'],
        hsv_v=CONFIG['hsv_v'],
        box=CONFIG['box'],
        cls=CONFIG['cls'],
        dfl=CONFIG['dfl'],
        workers=CONFIG['workers'],
        save_period=CONFIG['save_period'],
        verbose=CONFIG['verbose'],
        seed=CONFIG['seed'],
        deterministic=CONFIG['deterministic'],
    )

    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)

    # Print results
    print(f"\n📈 Final metrics:")
    print(f"   mAP50: {results.results_dict.get('metrics/mAP50(B)', 0):.3f}")
    print(f"   mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 0):.3f}")
    print(f"   Precision: {results.results_dict.get('metrics/precision(B)', 0):.3f}")
    print(f"   Recall: {results.results_dict.get('metrics/recall(B)', 0):.3f}")

    # Save path
    save_dir = Path(results.save_dir)
    print(f"\n💾 Model saved:")
    print(f"   Best: {save_dir / 'weights' / 'best.pt'}")
    print(f"   Last: {save_dir / 'weights' / 'last.pt'}")
    print(f"   Results: {save_dir / 'results.csv'}")

    # Validation
    print(f"\n🔍 Running validation on best model...")
    best_model = YOLO(save_dir / 'weights' / 'best.pt')
    val_results = best_model.val(data=CONFIG['data_yaml'])

    print(f"\n📊 Per-class metrics (best.pt):")
    class_names = data_config['names']
    for i, name in class_names.items():
        print(f"   {name}:")
        # Note: Ultralytics metrics are stored differently, adjust as needed
        print(f"      mAP50: {val_results.box.maps[i]:.3f}")
        print(f"      Precision: {val_results.box.p[i]:.3f}")
        print(f"      Recall: {val_results.box.r[i]:.3f}")

    print("\n✨ Done! Use best.pt for inference.")

if __name__ == '__main__':
    main()
