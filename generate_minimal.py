#!/usr/bin/env python3
"""
MINIMAL submission generator - only 3 smallest PDFs
For systems with very low memory
"""

import json
from pathlib import Path
from datetime import datetime
import sys

from enhanced_inference import EnhancedDocumentProcessor

def main():
    print("🔧 MINIMAL SUBMISSION GENERATOR (3 PDFs only)")
    print("=" * 50)

    # Find test PDFs
    test_dir = Path('data/test')
    all_pdfs = sorted([p for p in test_dir.glob('*.pdf') if 'Zone.Identifier' not in str(p)])

    # Sort by file size (smallest first)
    pdfs = sorted(all_pdfs, key=lambda p: p.stat().st_size)[:3]

    print(f"\n✓ Selected {len(pdfs)} smallest PDFs:")
    for pdf in pdfs:
        size_mb = pdf.stat().st_size / (1024 * 1024)
        print(f"   • {pdf.name} ({size_mb:.2f} MB)")

    submission = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'note': 'Minimal demo - 3 smallest PDFs only',
            'total_pdfs': len(pdfs),
            'model': 'best.pt',
            'conf_threshold': 0.25,
            'iou_threshold': 0.45,
            'dpi': 100  # Very low DPI for memory
        },
        'results': []
    }

    # Load model
    print("\n🤖 Loading model...")
    try:
        processor = EnhancedDocumentProcessor(
            model_path='best.pt',
            conf_threshold=0.25,
            iou_threshold=0.45
        )
        print("✓ Model loaded\n")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)

    # Process each PDF
    for i, pdf_path in enumerate(pdfs, 1):
        print(f"[{i}/{len(pdfs)}] 📄 {pdf_path.name}")

        try:
            results = processor.process_pdf(
                pdf_path=str(pdf_path),
                dpi=100,  # Very low DPI
                output_dir=None,
                generate_html=False,
                save_visualizations=False
            )

            results['relative_path'] = str(pdf_path.relative_to('data'))
            submission['results'].append(results)

            # Show summary
            summary = results.get('summary', {})
            print(f"   ✅ Pages: {results.get('total_pages', 0)}, "
                  f"Detections: {summary.get('total_detections', 0)}")

        except Exception as e:
            print(f"   ⚠️  Error: {e}")
            submission['results'].append({
                'pdf': pdf_path.name,
                'relative_path': str(pdf_path.relative_to('data')),
                'error': str(e),
                'status': 'failed'
            })

        # Save after each file
        with open('submission_minimal.json', 'w', encoding='utf-8') as f:
            json.dump(submission, f, indent=2, ensure_ascii=False)

        # Clean up
        import gc
        gc.collect()

    # Final statistics
    total_pages = sum(r.get('total_pages', 0) for r in submission['results'] if 'total_pages' in r)
    total_detections = sum(r.get('summary', {}).get('total_detections', 0) for r in submission['results'] if 'summary' in r)

    submission['metadata']['total_pages'] = total_pages
    submission['metadata']['total_detections'] = total_detections

    # Save final
    output = Path('submission_minimal.json')
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(submission, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 50)
    print("✅ DONE!")
    print(f"📁 Output: {output}")
    print(f"📊 Processed: {len(submission['results'])} PDFs (из 13 total)")
    print(f"📄 Pages: {total_pages}")
    print(f"🔍 Detections: {total_detections}")
    print(f"\n💡 Это демо с 3 файлами. Для полного - используй generate_simple.py")

if __name__ == '__main__':
    main()
