"""
Hybrid Inference Script - YOLO + OpenCV for QR Detection
Armeta CV Hackathon - Document Detection

Features:
- YOLO for signature, stamp, and QR (primary)
- OpenCV QRCodeDetector for QR (fallback/enhancement)
- Multi-scale inference (TTA)
- Batch processing
- JSON export
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
    'model_path': 'runs/detect/train/weights/best.pt',
    'conf_threshold': 0.25,
    'iou_threshold': 0.45,
    'imgsz': 1024,
    'augment': True,
    'device': 0,
    'use_opencv_qr': True,  # Enable hybrid QR detection
}

CLASS_NAMES = {
    0: 'signature',
    1: 'stamp',
    2: 'qr'
}

# ============= HYBRID DETECTION =============
def detect_opencv_qr(image):
    """
    Detect QR codes using OpenCV QRCodeDetector

    Args:
        image: numpy array (BGR format)

    Returns:
        list: QR detections with bboxes
    """
    qr_detector = cv2.QRCodeDetector()
    retval, decoded_info, points, _ = qr_detector.detectAndDecodeMulti(image)

    qr_detections = []

    if retval and points is not None:
        for i, qr_points in enumerate(points):
            # Convert polygon to bbox
            x_coords = qr_points[:, 0]
            y_coords = qr_points[:, 1]
            x1, y1 = float(x_coords.min()), float(y_coords.min())
            x2, y2 = float(x_coords.max()), float(y_coords.max())

            qr_detections.append({
                'class': 'qr',
                'class_id': 2,
                'confidence': 0.95,
                'bbox': {
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'width': x2 - x1,
                    'height': y2 - y1,
                },
                'source': 'opencv',
                'decoded': decoded_info[i] if decoded_info else None
            })

    return qr_detections

def is_duplicate_qr(det1, det2, iou_threshold=0.5):
    """Check if two QR detections overlap (to avoid duplicates)"""
    x1_inter = max(det1['bbox']['x1'], det2['bbox']['x1'])
    y1_inter = max(det1['bbox']['y1'], det2['bbox']['y1'])
    x2_inter = min(det1['bbox']['x2'], det2['bbox']['x2'])
    y2_inter = min(det1['bbox']['y2'], det2['bbox']['y2'])

    if x1_inter < x2_inter and y1_inter < y2_inter:
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        area1 = det1['bbox']['width'] * det1['bbox']['height']
        area2 = det2['bbox']['width'] * det2['bbox']['height']
        union_area = area1 + area2 - inter_area

        iou = inter_area / union_area if union_area > 0 else 0
        return iou > iou_threshold

    return False

def predict_image_hybrid(model, image_path, save_viz=True, output_dir='runs/detect/predict'):
    """
    Hybrid prediction: YOLO + OpenCV for QR

    Args:
        model: YOLO model
        image_path: Path to image
        save_viz: Save visualization
        output_dir: Output directory

    Returns:
        dict: Detection results
    """
    # Read image
    img = cv2.imread(str(image_path))

    # 1. YOLO prediction (all classes)
    results = model.predict(
        source=img,
        conf=CONFIG['conf_threshold'],
        iou=CONFIG['iou_threshold'],
        imgsz=CONFIG['imgsz'],
        augment=CONFIG['augment'],
        device=CONFIG['device'],
        save=save_viz,
        project=output_dir,
        verbose=False
    )

    # Extract YOLO detections
    detections = []
    yolo_qr_detections = []

    for r in results:
        boxes = r.boxes
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            det = {
                'class': CLASS_NAMES[cls_id],
                'class_id': cls_id,
                'confidence': float(boxes.conf[i]),
                'bbox': {
                    'x1': float(boxes.xyxy[i][0]),
                    'y1': float(boxes.xyxy[i][1]),
                    'x2': float(boxes.xyxy[i][2]),
                    'y2': float(boxes.xyxy[i][3]),
                    'width': float(boxes.xyxy[i][2] - boxes.xyxy[i][0]),
                    'height': float(boxes.xyxy[i][3] - boxes.xyxy[i][1]),
                },
                'source': 'yolo'
            }

            # Separate QR detections for duplicate check
            if cls_id == 2:
                yolo_qr_detections.append(det)

            detections.append(det)

    # 2. OpenCV QR detection (if enabled)
    if CONFIG['use_opencv_qr']:
        opencv_qr_detections = detect_opencv_qr(img)

        # Add non-duplicate OpenCV QRs
        for opencv_qr in opencv_qr_detections:
            is_dup = False
            for yolo_qr in yolo_qr_detections:
                if is_duplicate_qr(opencv_qr, yolo_qr):
                    is_dup = True
                    break

            if not is_dup:
                detections.append(opencv_qr)

    # Count detections
    count = {
        'signature': sum(1 for d in detections if d['class'] == 'signature'),
        'stamp': sum(1 for d in detections if d['class'] == 'stamp'),
        'qr': sum(1 for d in detections if d['class'] == 'qr'),
        'qr_yolo': len(yolo_qr_detections),
        'qr_opencv': sum(1 for d in detections if d['class'] == 'qr' and d.get('source') == 'opencv'),
        'total': len(detections)
    }

    return {
        'image': str(image_path),
        'image_size': {'width': img.shape[1], 'height': img.shape[0]},
        'detections': detections,
        'count': count
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
    stats = {
        'signature': 0, 'stamp': 0,
        'qr_total': 0, 'qr_yolo': 0, 'qr_opencv': 0,
        'total': 0
    }

    for img_path in tqdm(image_files, desc="Processing"):
        result = predict_image_hybrid(model, img_path, save_viz=save_viz)
        all_results.append(result)

        # Update stats
        stats['signature'] += result['count']['signature']
        stats['stamp'] += result['count']['stamp']
        stats['qr_total'] += result['count']['qr']
        stats['qr_yolo'] += result['count'].get('qr_yolo', 0)
        stats['qr_opencv'] += result['count'].get('qr_opencv', 0)
        stats['total'] += result['count']['total']

    # Save results
    output = {
        'model': f"{CONFIG['model_path']} (Hybrid: YOLO + OpenCV QR)",
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
    print(f"   QR codes (total): {stats['qr_total']}")
    print(f"     - From YOLO: {stats['qr_yolo']}")
    print(f"     - From OpenCV: {stats['qr_opencv']}")
    print(f"\n💾 Results saved to: {output_json}")

    return output

# ============= MAIN =============
def main():
    parser = argparse.ArgumentParser(description='Hybrid YOLOv8 + OpenCV Inference')
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
    parser.add_argument('--no-opencv', action='store_true',
                       help='Disable OpenCV QR detection (YOLO only)')

    args = parser.parse_args()

    # Update config
    CONFIG['model_path'] = args.model
    CONFIG['conf_threshold'] = args.conf
    CONFIG['use_opencv_qr'] = not args.no_opencv

    # Load model
    print(f"📦 Loading model: {CONFIG['model_path']}")
    model = YOLO(CONFIG['model_path'])
    print(f"✅ Model loaded")
    print(f"   Config: conf={CONFIG['conf_threshold']}, imgsz={CONFIG['imgsz']}, TTA={CONFIG['augment']}")
    print(f"   Hybrid QR: {CONFIG['use_opencv_qr']}")

    source_path = Path(args.source)

    # Single image
    if source_path.is_file():
        print(f"\n🖼️  Processing single image...")
        result = predict_image_hybrid(model, source_path, save_viz=not args.no_viz)

        # Save result
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"\n📊 Results:")
        print(f"   Total detections: {result['count']['total']}")
        print(f"   Signatures: {result['count']['signature']}")
        print(f"   Stamps: {result['count']['stamp']}")
        print(f"   QR codes: {result['count']['qr']} (YOLO: {result['count']['qr_yolo']}, OpenCV: {result['count']['qr_opencv']})")
        print(f"\n💾 Saved to: {args.output}")

    # Batch processing
    elif source_path.is_dir():
        print(f"\n📁 Processing folder...")
        predict_batch(model, source_path, output_json=args.output, save_viz=not args.no_viz)

    else:
        print(f"❌ Invalid source: {source_path}")

if __name__ == '__main__':
    main()
