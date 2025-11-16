#!/usr/bin/env python3
"""
Generate submission.json for hackathon

Processes all PDFs (data/pdf/ + data/test/) and creates final submission JSON
with all detection results including QR decoding and validation.

Usage:
    python generate_submission.py --output submission.json
"""

import argparse
import json
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

from enhanced_inference import EnhancedDocumentProcessor


def collect_all_pdfs():
    """Collect all PDF files from data/pdf/ and data/test/"""
    pdf_dirs = [
        Path('data/pdf'),
        Path('data/test')
    ]

    all_pdfs = []
    for pdf_dir in pdf_dirs:
        if pdf_dir.exists():
            pdfs = sorted(pdf_dir.glob('*.pdf'))
            # Filter out Zone.Identifier files
            pdfs = [p for p in pdfs if 'Zone.Identifier' not in str(p)]
            all_pdfs.extend(pdfs)

    return all_pdfs


def main():
    parser = argparse.ArgumentParser(
        description='Generate submission JSON for all PDFs'
    )
    parser.add_argument('--output', '-o', default='submission.json',
                       help='Output JSON file')
    parser.add_argument('--model', '-m', default='best.pt',
                       help='Path to model weights')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='NMS IoU threshold')
    parser.add_argument('--dpi', type=int, default=200,
                       help='PDF conversion DPI')

    args = parser.parse_args()

    print("=" * 70)
    print("📊 GENERATING SUBMISSION JSON")
    print("=" * 70)

    # Collect all PDFs
    all_pdfs = collect_all_pdfs()
    print(f"\n✓ Found {len(all_pdfs)} PDF files:")
    print(f"  - data/pdf/: {len([p for p in all_pdfs if 'data/pdf' in str(p)])}")
    print(f"  - data/test/: {len([p for p in all_pdfs if 'data/test' in str(p)])}")

    # Initialize processor
    print(f"\n🤖 Loading model: {args.model}")
    processor = EnhancedDocumentProcessor(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )
    print("✓ Model loaded\n")

    # Process all PDFs
    submission = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'total_pdfs': len(all_pdfs),
            'model': args.model,
            'conf_threshold': args.conf,
            'iou_threshold': args.iou,
            'dpi': args.dpi
        },
        'results': []
    }

    print("🔍 Processing PDFs...\n")

    for pdf_path in tqdm(all_pdfs, desc="Processing"):
        try:
            # Process PDF with enhanced features
            results = processor.process_pdf(
                pdf_path=str(pdf_path),
                validator_type='general',
                dpi=args.dpi
            )

            # Add relative path for easier identification
            results['relative_path'] = str(pdf_path.relative_to('data'))

            submission['results'].append(results)

        except Exception as e:
            print(f"\n⚠️  Error processing {pdf_path.name}: {e}")
            # Add error entry
            submission['results'].append({
                'pdf': pdf_path.name,
                'relative_path': str(pdf_path.relative_to('data')),
                'error': str(e),
                'status': 'failed'
            })

    # Calculate statistics
    total_pages = sum(r.get('total_pages', 0) for r in submission['results'] if 'total_pages' in r)
    total_detections = sum(r.get('summary', {}).get('total_detections', 0) for r in submission['results'] if 'summary' in r)

    submission['metadata']['total_pages'] = total_pages
    submission['metadata']['total_detections'] = total_detections

    # Count by class
    class_counts = {'signature': 0, 'stamp': 0, 'qr': 0}
    for result in submission['results']:
        if 'summary' in result and 'by_class' in result['summary']:
            for cls, count in result['summary']['by_class'].items():
                if cls in class_counts:
                    class_counts[cls] += count

    submission['metadata']['detections_by_class'] = class_counts

    # Count validation statuses
    validation_counts = {'valid': 0, 'warning': 0, 'invalid': 0}
    for result in submission['results']:
        if 'validation' in result:
            status = result['validation'].get('status', 'unknown')
            if status in validation_counts:
                validation_counts[status] += 1

    submission['metadata']['validation_summary'] = validation_counts

    # Save to file
    output_path = Path(args.output)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(submission, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("✅ SUBMISSION JSON CREATED")
    print("=" * 70)
    print(f"\n📁 Output: {output_path}")
    print(f"📊 Statistics:")
    print(f"   Total PDFs: {len(all_pdfs)}")
    print(f"   Total Pages: {total_pages}")
    print(f"   Total Detections: {total_detections}")
    print(f"\n   By Class:")
    for cls, count in class_counts.items():
        print(f"      {cls}: {count}")
    print(f"\n   Validation:")
    for status, count in validation_counts.items():
        print(f"      {status}: {count}")

    # File size
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n💾 File size: {file_size_mb:.2f} MB")

    print("\n✨ Ready for submission!")


if __name__ == '__main__':
    main()
