"""
Test YOLOv8 model on local test PDF documents

This version is for testing locally or in environments where you have
trained model weights available.

Usage:
    python test_model_local.py
"""

import cv2
import numpy as np
from pathlib import Path
from pdf2image import convert_from_path
from ultralytics import YOLO
import json
from PIL import Image, ImageDraw, ImageFont

# ============= CONFIGURATION =============
# Update these paths based on your environment
MODEL_PATH = 'runs/detect/train2/weights/best.pt'  # Local path
PDF_DIR = Path('data/test')
OUTPUT_DIR = Path('test_results')
DPI = 200

CLASS_NAMES = {0: 'signature', 1: 'stamp', 2: 'qr'}
COLORS = {
    0: (0, 255, 0),    # signature - green
    1: (255, 0, 0),    # stamp - blue
    2: (0, 0, 255),    # qr - red
}

# ============= FUNCTIONS =============
def draw_detections(image, results, conf_threshold=0.25):
    """Draw bounding boxes on image"""
    # Convert to PIL for better text rendering
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    detections = []

    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes

        for box in boxes:
            # Get box data
            conf = float(box.conf[0])
            if conf < conf_threshold:
                continue

            cls = int(box.cls[0])
            xyxy = box.xyxy[0].cpu().numpy()

            x1, y1, x2, y2 = map(int, xyxy)

            # Draw rectangle
            color = COLORS.get(cls, (255, 255, 255))
            # PIL uses RGB, swap for correct colors
            color_rgb = (color[2], color[1], color[0])

            draw.rectangle([x1, y1, x2, y2], outline=color_rgb, width=3)

            # Draw label
            label = f"{CLASS_NAMES[cls]} {conf:.2f}"

            # Draw text background
            bbox = draw.textbbox((x1, y1-20), label)
            draw.rectangle(bbox, fill=color_rgb)
            draw.text((x1, y1-20), label, fill=(255, 255, 255))

            detections.append({
                'class': CLASS_NAMES[cls],
                'confidence': conf,
                'bbox': [x1, y1, x2, y2]
            })

    # Convert back to OpenCV format
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR), detections


def test_pdf(pdf_path, model, output_dir):
    """Test model on single PDF"""
    print(f"\n📄 Processing: {pdf_path.name}")

    # Convert PDF to images
    try:
        images = convert_from_path(str(pdf_path), dpi=DPI)
    except Exception as e:
        print(f"   ❌ Error converting PDF: {e}")
        return None

    print(f"   Pages: {len(images)}")

    all_detections = []
    pdf_output_dir = output_dir / pdf_path.stem
    pdf_output_dir.mkdir(parents=True, exist_ok=True)

    for page_num, img in enumerate(images, 1):
        # Convert PIL to OpenCV format
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        h, w = img_cv.shape[:2]

        # Run inference
        results = model(img_cv, conf=0.25, verbose=False)

        # Draw detections
        img_annotated, detections = draw_detections(img_cv.copy(), results)

        # Save annotated image
        output_path = pdf_output_dir / f"page_{page_num}.jpg"
        cv2.imwrite(str(output_path), img_annotated)

        # Collect stats
        page_stats = {
            'page': page_num,
            'detections': len(detections),
            'objects': detections
        }
        all_detections.append(page_stats)

        # Print page summary
        if len(detections) > 0:
            counts = {}
            for det in detections:
                cls = det['class']
                counts[cls] = counts.get(cls, 0) + 1

            summary = ", ".join([f"{count} {cls}" for cls, count in counts.items()])
            print(f"   Page {page_num}: {summary}")
        else:
            print(f"   Page {page_num}: No detections")

    # Save JSON results
    result_json = {
        'pdf': pdf_path.name,
        'total_pages': len(images),
        'total_detections': sum(p['detections'] for p in all_detections),
        'pages': all_detections
    }

    json_path = pdf_output_dir / 'results.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result_json, f, indent=2, ensure_ascii=False)

    return result_json


def main():
    print("=" * 70)
    print("🧪 TESTING YOLOV8 MODEL ON LOCAL PDF DOCUMENTS")
    print("=" * 70)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"\n📦 Loading model: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
        print("   ✅ Model loaded successfully")
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        print(f"\n   Make sure the model file exists at: {Path(MODEL_PATH).absolute()}")
        return

    # Get PDF files
    if not PDF_DIR.exists():
        print(f"\n❌ PDF directory not found: {PDF_DIR.absolute()}")
        print(f"   Create the directory and add PDF files to test")
        return

    pdf_files = sorted(list(PDF_DIR.glob('*.pdf')))

    # Filter out Zone.Identifier files
    pdf_files = [p for p in pdf_files if 'Zone.Identifier' not in str(p)]

    if len(pdf_files) == 0:
        print(f"\n❌ No PDF files found in {PDF_DIR.absolute()}")
        return

    print(f"\n📂 Found {len(pdf_files)} PDF files to test")

    # Test each PDF
    all_results = []

    for pdf_path in pdf_files:
        result = test_pdf(pdf_path, model, OUTPUT_DIR)
        if result:
            all_results.append(result)

    # Summary statistics
    print("\n" + "=" * 70)
    print("📊 SUMMARY STATISTICS")
    print("=" * 70)

    total_pages = sum(r['total_pages'] for r in all_results)
    total_detections = sum(r['total_detections'] for r in all_results)

    print(f"Total PDFs tested: {len(all_results)}")
    print(f"Total pages: {total_pages}")
    print(f"Total detections: {total_detections}")
    if total_pages > 0:
        print(f"Average detections per page: {total_detections/total_pages:.1f}")

    # Count by class
    class_counts = {cls: 0 for cls in CLASS_NAMES.values()}

    for result in all_results:
        for page in result['pages']:
            for obj in page['objects']:
                class_counts[obj['class']] += 1

    print("\nDetections by class:")
    for cls, count in class_counts.items():
        print(f"   {cls}: {count}")

    # Save summary
    summary = {
        'total_pdfs': len(all_results),
        'total_pages': total_pages,
        'total_detections': total_detections,
        'class_counts': class_counts,
        'pdfs': all_results
    }

    summary_path = OUTPUT_DIR / 'summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Results saved to: {OUTPUT_DIR.absolute()}")
    print(f"   - Annotated images: {OUTPUT_DIR}/<pdf_name>/page_*.jpg")
    print(f"   - Summary: {summary_path}")
    print("\n🎉 Testing complete!")


if __name__ == '__main__':
    main()
