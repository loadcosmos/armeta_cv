"""
Enhanced Production Inference System

Combines:
- YOLO detection
- QR code decoding
- Document validation
- HTML report generation

This is THE killer feature that will win the hackathon!

Usage:
    python enhanced_inference.py --input document.pdf --output results/ --html
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from pdf2image import convert_from_path
from typing import List, Dict
from tqdm import tqdm
import json

from inference import DocumentDetector, find_model
from qr_decoder import QRDecoder
from validator import DocumentValidator, ContractValidator, LicenseValidator, format_validation_report
from html_reporter import HTMLReporter


class EnhancedDocumentProcessor:
    """
    Complete document processing pipeline with all features
    """

    def __init__(self, model_path: str = None,
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45,
                 validator_type: str = 'default'):
        """
        Initialize enhanced processor

        Args:
            model_path: Path to YOLO model
            conf_threshold: Detection confidence threshold
            iou_threshold: NMS IoU threshold
            validator_type: Type of validator ('default', 'contract', 'license')
        """
        # Initialize components
        self.detector = DocumentDetector(
            model_path=model_path or find_model(),
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold
        )

        self.qr_decoder = QRDecoder()

        # Select validator
        if validator_type == 'contract':
            self.validator = ContractValidator()
        elif validator_type == 'license':
            self.validator = LicenseValidator()
        else:
            self.validator = DocumentValidator()

        self.html_reporter = HTMLReporter()

    def process_pdf(self, pdf_path: str, dpi: int = 200,
                    output_dir: Path = None,
                    generate_html: bool = True,
                    save_visualizations: bool = True) -> Dict:
        """
        Complete processing pipeline for PDF

        Args:
            pdf_path: Path to PDF file
            dpi: Resolution for PDF conversion
            output_dir: Output directory
            generate_html: Generate HTML report
            save_visualizations: Save annotated images

        Returns:
            Complete results with detections, QR data, and validation
        """
        pdf_path = Path(pdf_path)
        print(f"\n📄 Processing: {pdf_path.name}")

        # Convert PDF to images
        print("  Converting PDF to images...")
        pil_images = convert_from_path(str(pdf_path), dpi=dpi)

        results = {
            'pdf': pdf_path.name,
            'total_pages': len(pil_images),
            'pages': []
        }

        annotated_images = []

        # Process each page
        for page_num, pil_img in enumerate(tqdm(pil_images, desc="  Processing pages"), 1):
            # Convert to OpenCV format
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            # 1. Detect objects
            detections = self.detector.detect_image(img)

            # 2. Decode QR codes
            detections = self.qr_decoder.decode_all_qr_codes(img, detections)

            # 3. Visualize
            if save_visualizations or generate_html:
                img_vis = self.detector.visualize_detections(img, detections)

                # Add QR data to visualization
                for det in detections:
                    if det.get('class') == 'qr' and 'qr_data' in det:
                        qr_data = det['qr_data']
                        if qr_data.get('data'):
                            bbox = det['bbox']
                            x1, y1 = int(bbox['x1']), int(bbox['y1'])

                            # Draw QR data
                            data_text = qr_data['data'][:50]
                            cv2.putText(img_vis, f"QR: {data_text}",
                                       (x1, y1 - 30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                annotated_images.append(img_vis)

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

        # Calculate overall summary
        total_detections = sum(len(p['detections']) for p in results['pages'])
        overall_counts = {}
        for page in results['pages']:
            for cls, count in page['summary'].items():
                overall_counts[cls] = overall_counts.get(cls, 0) + count

        results['summary'] = {
            'total_detections': total_detections,
            'by_class': overall_counts
        }

        # 4. Validate document
        print("  Validating document...")
        validation = self.validator.validate(results)
        results['validation'] = validation

        # Print validation results
        print(f"\n{format_validation_report(validation)}\n")

        # 5. Save results
        if output_dir:
            output_dir = Path(output_dir)
            pdf_output = output_dir / pdf_path.stem
            pdf_output.mkdir(parents=True, exist_ok=True)

            # Save JSON
            json_path = pdf_output / 'results.json'
            with open(json_path, 'w', encoding='utf-8') as f:
                # Create serializable copy (remove OpenCV objects)
                serializable_results = self._make_serializable(results)
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)

            print(f"  ✓ Saved JSON: {json_path}")

            # Save visualizations
            if save_visualizations:
                for i, img in enumerate(annotated_images, 1):
                    img_path = pdf_output / f'page_{i}.jpg'
                    cv2.imwrite(str(img_path), img)
                print(f"  ✓ Saved {len(annotated_images)} visualizations")

            # Generate HTML report
            if generate_html:
                html_path = pdf_output / 'report.html'
                self.html_reporter.generate_report(
                    results,
                    validation=validation,
                    page_images=annotated_images,
                    output_path=html_path
                )
                print(f"  ✓ Generated HTML report: {html_path}")

        return results

    def _make_serializable(self, obj):
        """Convert object to JSON-serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def process_batch(self, input_dir: Path,
                     output_dir: Path,
                     generate_html: bool = True) -> List[Dict]:
        """
        Process multiple PDF files

        Args:
            input_dir: Directory containing PDFs
            output_dir: Output directory
            generate_html: Generate HTML reports

        Returns:
            List of results
        """
        pdf_files = list(input_dir.glob('*.pdf'))
        pdf_files = [p for p in pdf_files if 'Zone.Identifier' not in str(p)]

        print(f"\n📂 Found {len(pdf_files)} PDF files")

        all_results = []

        for pdf_path in pdf_files:
            try:
                result = self.process_pdf(
                    pdf_path,
                    output_dir=output_dir,
                    generate_html=generate_html
                )
                all_results.append(result)
            except Exception as e:
                print(f"  ❌ Error processing {pdf_path.name}: {e}")

        # Batch validation summary
        batch_validation = self.validator.validate_batch(all_results)

        # Save batch summary
        summary_path = output_dir / 'batch_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(batch_validation, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Batch processing complete!")
        print(f"   Total: {batch_validation['total_documents']}")
        print(f"   Valid: {batch_validation['valid']} ({batch_validation['pass_rate']:.1%})")
        print(f"   Warnings: {batch_validation['warnings']}")
        print(f"   Invalid: {batch_validation['invalid']}")

        return all_results


def main():
    parser = argparse.ArgumentParser(
        description='Enhanced document detection with QR decoding and validation'
    )
    parser.add_argument('--input', '-i', required=True,
                       help='Input PDF file or directory')
    parser.add_argument('--output', '-o', default='enhanced_results',
                       help='Output directory')
    parser.add_argument('--model', '-m', default=None,
                       help='Path to model weights')
    parser.add_argument('--html', action='store_true',
                       help='Generate HTML report')
    parser.add_argument('--no-viz', action='store_true',
                       help='Skip visualization images')
    parser.add_argument('--batch', '-b', action='store_true',
                       help='Process directory of PDFs')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='NMS IoU threshold')
    parser.add_argument('--validator', choices=['default', 'contract', 'license'],
                       default='default', help='Validator type')
    parser.add_argument('--dpi', type=int, default=200,
                       help='PDF conversion DPI')

    args = parser.parse_args()

    # Initialize processor
    print("🚀 Enhanced Document Processing System")
    print("=" * 70)
    print(f"Model: {args.model or 'auto-detect'}")
    print(f"Validator: {args.validator}")
    print(f"Confidence: {args.conf}")
    print(f"NMS IoU: {args.iou}")
    print()

    processor = EnhancedDocumentProcessor(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        validator_type=args.validator
    )

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process input
    input_path = Path(args.input)

    if args.batch or input_path.is_dir():
        # Batch processing
        processor.process_batch(
            input_path,
            output_dir,
            generate_html=args.html
        )
    else:
        # Single file
        if not input_path.exists():
            print(f"❌ Error: File not found: {input_path}")
            return

        processor.process_pdf(
            input_path,
            output_dir=output_dir,
            generate_html=args.html,
            save_visualizations=not args.no_viz
        )

    print(f"\n✅ Results saved to: {output_dir.absolute()}")


if __name__ == '__main__':
    main()
