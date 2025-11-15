"""
YOLOv8 Inference Script - Optimized for Small Objects
Armeta CV Hackathon - Document Detection

Features:
- Multi-scale inference (TTA)
- Confidence threshold optimization
- Batch processing
- Results export to JSON
"""

from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import argparse

# ============= CONFIGURATION =============
CONFIG = {
    'model_path': 'runs/detect/train/weights/best.pt',  # Update with your path
    'conf_threshold': 0.25,  # Lower than default (0.25 vs 0.3) for better recall
    'iou_threshold': 0.45,   # NMS threshold
    'imgsz': 1024,          # Must match training size!
    'augment': True,        # Test-time augmentation (TTA)
    'device': 0,            # GPU
}

CLASS_NAMES = {
    0: 'signature',
    1: 'stamp',
    2: 'qr'
}

# ============= FUNCTIONS =============
def predict_image(model, image_path, save_viz=True, output_dir='runs/detect/predict'):
    """
    Predict on a single image with optimized settings

    Args:
        model: YOLO model
        image_path: Path to image
        save_viz: Save visualization
        output_dir: Output directory

    Returns:
        dict: Detection results
    """
    results = model.predict(
        source=image_path,
        conf=CONFIG['conf_threshold'],
        iou=CONFIG['iou_threshold'],
        imgsz=CONFIG['imgsz'],
        augment=CONFIG['augment'],  # ⭐ TTA for better small object detection
        device=CONFIG['device'],
        save=save_viz,
        project=output_dir,
        verbose=False
    )

    # Extract detections
    detections = []
    for r in results:
        boxes = r.boxes
        for i in range(len(boxes)):
            det = {
                'class': CLASS_NAMES[int(boxes.cls[i])],
                'class_id': int(boxes.cls[i]),
                'confidence': float(boxes.conf[i]),
                'bbox': {
                    'x1': float(boxes.xyxy[i][0]),
                    'y1': float(boxes.xyxy[i][1]),
                    'x2': float(boxes.xyxy[i][2]),
                    'y2': float(boxes.xyxy[i][3]),
                    'width': float(boxes.xyxy[i][2] - boxes.xyxy[i][0]),
                    'height': float(boxes.xyxy[i][3] - boxes.xyxy[i][1]),
                }
            }
            detections.append(det)

    return {
        'image': str(image_path),
        'image_size': {'width': r.orig_shape[1], 'height': r.orig_shape[0]},
        'detections': detections,
        'count': {
            'signature': sum(1 for d in detections if d['class'] == 'signature'),
            'stamp': sum(1 for d in detections if d['class'] == 'stamp'),
            'qr': sum(1 for d in detections if d['class'] == 'qr'),
            'total': len(detections)
        }
    }

def predict_batch(model, image_folder, output_json='results.json', save_viz=True):
    """
    Predict on all images in a folder

    Args:
        model: YOLO model
        image_folder: Folder with images
        output_json: Output JSON file
        save_viz: Save visualizations

    Returns:
        dict: All results
    """
    image_folder = Path(image_folder)
    image_files = list(image_folder.glob('*.png')) + list(image_folder.glob('*.jpg'))

    print(f"📁 Found {len(image_files)} images in {image_folder}")

    all_results = []
    stats = {'signature': 0, 'stamp': 0, 'qr': 0, 'total': 0}

    for img_path in tqdm(image_files, desc="Processing"):
        result = predict_image(model, img_path, save_viz=save_viz)
        all_results.append(result)

        # Update stats
        stats['signature'] += result['count']['signature']
        stats['stamp'] += result['count']['stamp']
        stats['qr'] += result['count']['qr']
        stats['total'] += result['count']['total']

    # Save results
    output = {
        'model': CONFIG['model_path'],
        'config': CONFIG,
        'stats': stats,
        'results': all_results
    }

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n📊 Statistics:")
    print(f"   Images processed: {len(image_files)}")
    print(f"   Total detections: {stats['total']}")
    print(f"   Signatures: {stats['signature']}")
    print(f"   Stamps: {stats['stamp']}")
    print(f"   QR codes: {stats['qr']}")
    print(f"\n💾 Results saved to: {output_json}")

    return output

def compare_thresholds(model, image_path, thresholds=[0.15, 0.20, 0.25, 0.30, 0.35]):
    """
    Test different confidence thresholds to find optimal value

    Args:
        model: YOLO model
        image_path: Test image path
        thresholds: List of thresholds to test
    """
    print(f"\n🔍 Testing confidence thresholds on: {image_path}")
    print("-" * 60)

    for conf in thresholds:
        CONFIG['conf_threshold'] = conf
        result = predict_image(model, image_path, save_viz=False)

        print(f"conf={conf:.2f} → Detections: {result['count']['total']} "
              f"(sig:{result['count']['signature']}, "
              f"stamp:{result['count']['stamp']}, "
              f"qr:{result['count']['qr']})")

    print("-" * 60)
    print("💡 Recommendation: Lower conf (0.20-0.25) for better recall on small objects")

# ============= MAIN =============
def main():
    parser = argparse.ArgumentParser(description='YOLOv8 Inference for Document Detection')
    parser.add_argument('--model', type=str, default=CONFIG['model_path'],
                       help='Path to model weights')
    parser.add_argument('--source', type=str, required=True,
                       help='Image file or folder path')
    parser.add_argument('--conf', type=float, default=CONFIG['conf_threshold'],
                       help='Confidence threshold')
    parser.add_argument('--output', type=str, default='results.json',
                       help='Output JSON file')
    parser.add_argument('--no-viz', action='store_true',
                       help='Disable visualization')
    parser.add_argument('--test-thresholds', action='store_true',
                       help='Test different confidence thresholds')

    args = parser.parse_args()

    # Update config
    CONFIG['model_path'] = args.model
    CONFIG['conf_threshold'] = args.conf

    # Load model
    print(f"📦 Loading model: {CONFIG['model_path']}")
    model = YOLO(CONFIG['model_path'])
    print(f"✅ Model loaded")
    print(f"   Config: conf={CONFIG['conf_threshold']}, imgsz={CONFIG['imgsz']}, TTA={CONFIG['augment']}")

    source_path = Path(args.source)

    # Test thresholds mode
    if args.test_thresholds:
        if source_path.is_file():
            compare_thresholds(model, source_path)
        else:
            print("❌ --test-thresholds requires a single image file")
        return

    # Single image
    if source_path.is_file():
        print(f"\n🖼️  Processing single image...")
        result = predict_image(model, source_path, save_viz=not args.no_viz)

        # Save result
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"\n📊 Results:")
        print(f"   Total detections: {result['count']['total']}")
        print(f"   Signatures: {result['count']['signature']}")
        print(f"   Stamps: {result['count']['stamp']}")
        print(f"   QR codes: {result['count']['qr']}")
        print(f"\n💾 Saved to: {args.output}")

    # Batch processing
    elif source_path.is_dir():
        print(f"\n📁 Processing folder...")
        predict_batch(model, source_path, output_json=args.output, save_viz=not args.no_viz)

    else:
        print(f"❌ Invalid source: {source_path}")

if __name__ == '__main__':
    main()
