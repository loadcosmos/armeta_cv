"""

Test YOLOv8 model with optimized settings for overlapping objects

 

This version allows detection of overlapping signatures and stamps

by reducing NMS IoU threshold.

 

Usage:

    python test_model_overlapping.py

"""

 

import cv2

import numpy as np

from pathlib import Path

from pdf2image import convert_from_path

from ultralytics import YOLO

import json

from PIL import Image, ImageDraw, ImageFont

 

# ============= CONFIGURATION =============

MODEL_PATH = 'runs/detect/train2/weights/best.pt'

PDF_DIR = Path('data/test')

OUTPUT_DIR = Path('test_results_overlapping')

DPI = 200

 

# Detection settings for overlapping objects

CONF_THRESHOLD = 0.20  # Lower confidence to catch more objects

IOU_THRESHOLD = 0.3    # Lower IoU allows more overlapping boxes (default: 0.45)

MAX_DET = 300          # Maximum detections per image

 

CLASS_NAMES = {0: 'signature', 1: 'stamp', 2: 'qr'}

COLORS = {

    0: (0, 255, 0),    # signature - green

    1: (255, 0, 0),    # stamp - blue

    2: (0, 0, 255),    # qr - red

}

 

# ============= FUNCTIONS =============

def draw_detections(image, results, conf_threshold=0.20):

    """Draw bounding boxes with transparency for overlapping objects"""

    # Convert to PIL for better rendering

    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

 

    # Create overlay for semi-transparent boxes

    overlay = Image.new('RGBA', pil_img.size, (255, 255, 255, 0))

    draw = ImageDraw.Draw(overlay)

 

    detections = []

 

    if len(results) > 0 and results[0].boxes is not None:

        boxes = results[0].boxes

 

        # Sort boxes by confidence (draw lower confidence first)

        sorted_indices = sorted(range(len(boxes)), key=lambda i: float(boxes[i].conf[0]))

 

        for idx in sorted_indices:

            box = boxes[idx]

 

            # Get box data

            conf = float(box.conf[0])

            if conf < conf_threshold:

                continue

 

            cls = int(box.cls[0])

            xyxy = box.xyxy[0].cpu().numpy()

 

            x1, y1, x2, y2 = map(int, xyxy)

 

            # Draw semi-transparent rectangle

            color = COLORS.get(cls, (255, 255, 255))

            color_rgb = (color[2], color[1], color[0])

 

            # Semi-transparent fill

            color_rgba = (*color_rgb, 60)  # 60/255 transparency

            draw.rectangle([x1, y1, x2, y2], fill=color_rgba, outline=color_rgb, width=3)

 

            detections.append({

                'class': CLASS_NAMES[cls],

                'confidence': conf,

                'bbox': [x1, y1, x2, y2]

            })

 

    # Composite overlay onto base image

    pil_img = pil_img.convert('RGBA')

    pil_img = Image.alpha_composite(pil_img, overlay)

 

    # Draw labels on top

    draw = ImageDraw.Draw(pil_img)

    for det in detections:

        x1, y1, x2, y2 = det['bbox']

        cls_name = det['class']

        conf = det['confidence']

 

        # Get color

        cls_id = [k for k, v in CLASS_NAMES.items() if v == cls_name][0]

        color = COLORS.get(cls_id, (255, 255, 255))

        color_rgb = (color[2], color[1], color[0])

 

        # Draw label

        label = f"{cls_name} {conf:.2f}"

        bbox = draw.textbbox((x1, y1-20), label)

        draw.rectangle(bbox, fill=color_rgb)

        draw.text((x1, y1-20), label, fill=(255, 255, 255))

 

    # Convert back to OpenCV format

    pil_img = pil_img.convert('RGB')

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR), detections

 

 

def test_pdf(pdf_path, model, output_dir):

    """Test model on single PDF with overlapping detection"""

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

 

        # Run inference with optimized settings for overlapping objects

        results = model(

            img_cv,

            conf=CONF_THRESHOLD,

            iou=IOU_THRESHOLD,      # Lower IoU allows overlapping boxes

            max_det=MAX_DET,

            verbose=False

        )

 

        # Draw detections

        img_annotated, detections = draw_detections(img_cv.copy(), results, conf_threshold=CONF_THRESHOLD)

 

        # Save annotated image

        output_path = pdf_output_dir / f"page_{page_num}.jpg"

        cv2.imwrite(str(output_path), img_annotated)

 

        # Check for overlapping objects

        overlapping_pairs = []

        for i, det1 in enumerate(detections):

            for j, det2 in enumerate(detections[i+1:], i+1):

                # Calculate IoU

                x1_1, y1_1, x2_1, y2_1 = det1['bbox']

                x1_2, y1_2, x2_2, y2_2 = det2['bbox']

 

                # Intersection

                xi1, yi1 = max(x1_1, x1_2), max(y1_1, y1_2)

                xi2, yi2 = min(x2_1, x2_2), min(y2_1, y2_2)

 

                if xi2 > xi1 and yi2 > yi1:

                    inter_area = (xi2 - xi1) * (yi2 - yi1)

                    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)

                    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

                    union_area = box1_area + box2_area - inter_area

                    iou = inter_area / union_area if union_area > 0 else 0

 

                    if iou > 0.1:  # Overlapping threshold

                        overlapping_pairs.append({

                            'object1': f"{det1['class']} ({det1['confidence']:.2f})",

                            'object2': f"{det2['class']} ({det2['confidence']:.2f})",

                            'iou': iou

                        })

 

        # Collect stats

        page_stats = {

            'page': page_num,

            'detections': len(detections),

            'overlapping_pairs': overlapping_pairs,

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

            overlap_info = f" [{len(overlapping_pairs)} overlapping]" if overlapping_pairs else ""

            print(f"   Page {page_num}: {summary}{overlap_info}")

        else:

            print(f"   Page {page_num}: No detections")

 

    # Save JSON results

    result_json = {

        'pdf': pdf_path.name,

        'total_pages': len(images),

        'total_detections': sum(p['detections'] for p in all_detections),

        'total_overlapping': sum(len(p['overlapping_pairs']) for p in all_detections),

        'pages': all_detections

    }

 

    json_path = pdf_output_dir / 'results.json'

    with open(json_path, 'w', encoding='utf-8') as f:

        json.dump(result_json, f, indent=2, ensure_ascii=False)

 

    return result_json

 

 

def main():

    print("=" * 70)

    print("🧪 TESTING WITH OVERLAPPING OBJECT DETECTION")

    print("=" * 70)

    print(f"Settings:")

    print(f"  - Confidence threshold: {CONF_THRESHOLD}")

    print(f"  - NMS IoU threshold: {IOU_THRESHOLD} (lower = more overlapping boxes)")

    print(f"  - Max detections: {MAX_DET}")

    print()

 

    # Create output directory

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

 

    # Load model

    print(f"📦 Loading model: {MODEL_PATH}")

    try:

        model = YOLO(MODEL_PATH)

        print("   ✅ Model loaded successfully")

    except Exception as e:

        print(f"   ❌ Error loading model: {e}")

        return

 

    # Get PDF files

    if not PDF_DIR.exists():

        print(f"\n❌ PDF directory not found: {PDF_DIR.absolute()}")

        return

 

    pdf_files = sorted(list(PDF_DIR.glob('*.pdf')))

    pdf_files = [p for p in pdf_files if 'Zone.Identifier' not in str(p)]

 

    if len(pdf_files) == 0:

        print(f"\n❌ No PDF files found in {PDF_DIR.absolute()}")

        return

 

    print(f"📂 Found {len(pdf_files)} PDF files to test")

 

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

    total_overlapping = sum(r['total_overlapping'] for r in all_results)

 

    print(f"Total PDFs tested: {len(all_results)}")

    print(f"Total pages: {total_pages}")

    print(f"Total detections: {total_detections}")

    print(f"Total overlapping pairs: {total_overlapping}")

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

        'settings': {

            'conf_threshold': CONF_THRESHOLD,

            'iou_threshold': IOU_THRESHOLD,

            'max_det': MAX_DET

        },

        'total_pdfs': len(all_results),

        'total_pages': total_pages,

        'total_detections': total_detections,

        'total_overlapping': total_overlapping,

        'class_counts': class_counts,

        'pdfs': all_results

    }

 

    summary_path = OUTPUT_DIR / 'summary.json'

    with open(summary_path, 'w', encoding='utf-8') as f:

        json.dump(summary, f, indent=2, ensure_ascii=False)

 

    print(f"\n✅ Results saved to: {OUTPUT_DIR.absolute()}")

    print(f"   - Annotated images: {OUTPUT_DIR}/<pdf_name>/page_*.jpg")

    print(f"   - Summary: {summary_path}")

    print("\n💡 TIP: Semi-transparent boxes show overlapping detections")

    print("🎉 Testing complete!")

 

 

if __name__ == '__main__':

    main()