"""
Production-ready inference script for document object detection

Usage:
    # Single PDF
    python inference.py --input document.pdf --output results/

    # Batch processing
    python inference.py --input data/pdfs/ --output results/ --batch

    # With visualization
    python inference.py --input doc.pdf --output results/ --visualize
"""

import argparse
import cv2
import json
import numpy as np
from pathlib import Path
from pdf2image import convert_from_path
from ultralytics import YOLO
from datetime import datetime
from typing import List, Dict, Tuple
from tqdm import tqdm

# Default model path
DEFAULT_MODEL = 'runs/detect/train2/weights/best.pt'

CLASS_NAMES = {0: 'signature', 1: 'stamp', 2: 'qr'}
COLORS = {
    0: (0, 255, 0),    # signature - green
    1: (255, 0, 0),    # stamp - blue
    2: (0, 0, 255),    # qr - red
}


class DocumentDetector:
    """Document object detection wrapper"""

    def __init__(self, model_path: str = DEFAULT_MODEL,
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45):
        """
        Initialize detector

        Args:
            model_path: Path to YOLOv8 weights
            conf_threshold: Confidence threshold for detections
            iou_threshold: NMS IoU threshold
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def detect_image(self, image: np.ndarray) -> List[Dict]:
        """
        Detect objects in single image

        Args:
            image: OpenCV image (BGR)

        Returns:
            List of detections with class, confidence, bbox
        """
        results = self.model(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )

        detections = []

        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes

            for box in boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                xyxy = box.xyxy[0].cpu().numpy()

                x1, y1, x2, y2 = map(float, xyxy)

                detections.append({
                    'class': CLASS_NAMES[cls],
                    'class_id': cls,
                    'confidence': conf,
                    'bbox': {
                        'x1': x1, 'y1': y1,
                        'x2': x2, 'y2': y2,
                        'width': x2 - x1,
                        'height': y2 - y1
                    }
                })

        return detections

    def detect_pdf(self, pdf_path: str, dpi: int = 200) -> Dict:
        """
        Detect objects in PDF document

        Args:
            pdf_path: Path to PDF file
            dpi: Resolution for PDF conversion

        Returns:
            Detection results for all pages
        """
        # Convert PDF to images
        images = convert_from_path(pdf_path, dpi=dpi)

        results = {
            'pdf': Path(pdf_path).name,
            'timestamp': datetime.now().isoformat(),
            'total_pages': len(images),
            'pages': []
        }

        for page_num, pil_img in enumerate(images, 1):
            # Convert to OpenCV format
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            # Detect objects
            detections = self.detect_image(img)

            # Count by class
            class_counts = {}
            for det in detections:
                cls = det['class']
                class_counts[cls] = class_counts.get(cls, 0) + 1

            results['pages'].append({
                'page': page_num,
                'image_size': {'width': img.shape[1], 'height': img.shape[0]},
                'detections': detections,
                'summary': class_counts
            })

        # Overall summary
        total_detections = sum(len(p['detections']) for p in results['pages'])
        overall_counts = {}
        for page in results['pages']:
            for cls, count in page['summary'].items():
                overall_counts[cls] = overall_counts.get(cls, 0) + count

        results['summary'] = {
            'total_detections': total_detections,
            'by_class': overall_counts
        }

        return results

    def visualize_detections(self, image: np.ndarray,
                            detections: List[Dict]) -> np.ndarray:
        """
        Draw detections on image

        Args:
            image: OpenCV image (BGR)
            detections: List of detections

        Returns:
            Annotated image
        """
        img_vis = image.copy()

        for det in detections:
            bbox = det['bbox']
            x1, y1 = int(bbox['x1']), int(bbox['y1'])
            x2, y2 = int(bbox['x2']), int(bbox['y2'])

            cls_id = det['class_id']
            color = COLORS.get(cls_id, (255, 255, 255))

            # Draw rectangle
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{det['class']} {det['confidence']:.2f}"

            # Text background
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(img_vis, (x1, y1 - text_h - 10),
                         (x1 + text_w, y1), color, -1)

            # Text
            cv2.putText(img_vis, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return img_vis


def process_single_pdf(detector: DocumentDetector,
                      pdf_path: Path,
                      output_dir: Path,
                      visualize: bool = False,
                      dpi: int = 200) -> Dict:
    """Process single PDF file"""

    print(f"Processing: {pdf_path.name}")

    # Run detection
    results = detector.detect_pdf(str(pdf_path), dpi=dpi)

    # Create output directory
    pdf_output = output_dir / pdf_path.stem
    pdf_output.mkdir(parents=True, exist_ok=True)

    # Save JSON results
    json_path = pdf_output / 'results.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Visualize if requested
    if visualize:
        images = convert_from_path(str(pdf_path), dpi=dpi)

        for page_idx, pil_img in enumerate(images):
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            detections = results['pages'][page_idx]['detections']

            img_vis = detector.visualize_detections(img, detections)

            vis_path = pdf_output / f'page_{page_idx + 1}.jpg'
            cv2.imwrite(str(vis_path), img_vis)

    # Print summary
    summary = results['summary']
    print(f"  ✓ {summary['total_detections']} detections")
    for cls, count in summary['by_class'].items():
        print(f"    - {cls}: {count}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Document object detection inference'
    )
    parser.add_argument('--input', '-i', required=True,
                       help='Input PDF file or directory')
    parser.add_argument('--output', '-o', default='inference_results',
                       help='Output directory')
    parser.add_argument('--model', '-m', default=DEFAULT_MODEL,
                       help='Path to model weights')
    parser.add_argument('--visualize', '-v', action='store_true',
                       help='Save visualization images')
    parser.add_argument('--batch', '-b', action='store_true',
                       help='Process directory of PDFs')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='NMS IoU threshold')
    parser.add_argument('--dpi', type=int, default=200,
                       help='PDF conversion DPI')

    args = parser.parse_args()

    # Initialize detector
    print(f"Loading model: {args.model}")
    detector = DocumentDetector(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )
    print("✓ Model loaded\n")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process input
    input_path = Path(args.input)

    if args.batch or input_path.is_dir():
        # Batch processing
        pdf_files = list(input_path.glob('*.pdf'))
        pdf_files = [p for p in pdf_files if 'Zone.Identifier' not in str(p)]

        print(f"Found {len(pdf_files)} PDF files\n")

        all_results = []
        for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
            result = process_single_pdf(
                detector, pdf_path, output_dir,
                visualize=args.visualize, dpi=args.dpi
            )
            all_results.append(result)

        # Save batch summary
        summary = {
            'total_pdfs': len(all_results),
            'total_pages': sum(r['total_pages'] for r in all_results),
            'total_detections': sum(r['summary']['total_detections'] for r in all_results),
            'pdfs': [r['pdf'] for r in all_results]
        }

        summary_path = output_dir / 'batch_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Processed {len(all_results)} PDFs")
        print(f"✓ Total detections: {summary['total_detections']}")

    else:
        # Single file processing
        if not input_path.exists():
            print(f"Error: File not found: {input_path}")
            return

        result = process_single_pdf(
            detector, input_path, output_dir,
            visualize=args.visualize, dpi=args.dpi
        )

    print(f"\n✓ Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
