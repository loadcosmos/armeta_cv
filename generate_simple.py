#!/usr/bin/env python3
"""
Lightweight submission generator - processes one PDF at a time
Safe for low memory systems
"""

import json
from pathlib import Path
from datetime import datetime
import sys

# Minimal imports
from enhanced_inference import EnhancedDocumentProcessor

def main():
    print("🔧 LIGHTWEIGHT SUBMISSION GENERATOR")
    print("=" * 50)

    # Find test PDFs
    test_dir = Path('data/test')
    pdfs = sorted([p for p in test_dir.glob('*.pdf') if 'Zone.Identifier' not in str(p)])

    print(f"\n✓ Found {len(pdfs)} PDF files")
    print("\nProcessing one at a time to save memory...")

    # Load results if exists
    output_file = Path('submission_progress.json')
    if output_file.exists():
        print(f"\n📂 Loading previous progress from {output_file}")
        with open(output_file, 'r', encoding='utf-8') as f:
            submission = json.load(f)
        processed_files = {r['pdf'] for r in submission['results'] if 'pdf' in r}
        print(f"✓ Already processed: {len(processed_files)} files")
    else:
        submission = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_pdfs': len(pdfs),
                'model': 'best.pt',
                'conf_threshold': 0.25,
                'iou_threshold': 0.45,
                'dpi': 150  # Lower DPI for memory
            },
            'results': []
        }
        processed_files = set()

    # Initialize processor (only once)
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

    # Process PDFs one by one
    for i, pdf_path in enumerate(pdfs, 1):
        pdf_name = pdf_path.name

        # Skip if already processed
        if pdf_name in processed_files:
            print(f"[{i}/{len(pdfs)}] ⏭️  Skipping {pdf_name} (already done)")
            continue

        print(f"[{i}/{len(pdfs)}] 📄 Processing: {pdf_name}")

        try:
            # Process with minimal options (no HTML, no viz)
            results = processor.process_pdf(
                pdf_path=str(pdf_path),
                dpi=150,  # Lower DPI = less memory
                output_dir=None,
                generate_html=False,
                save_visualizations=False
            )

            # Add to results
            results['relative_path'] = str(pdf_path.relative_to('data'))
            submission['results'].append(results)

            # Save progress immediately
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(submission, f, indent=2, ensure_ascii=False)

            print(f"   ✅ Success ({results.get('total_pages', '?')} pages)")

        except Exception as e:
            print(f"   ⚠️  Error: {e}")
            # Add error entry
            submission['results'].append({
                'pdf': pdf_name,
                'relative_path': str(pdf_path.relative_to('data')),
                'error': str(e),
                'status': 'failed'
            })
            # Save even errors
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(submission, f, indent=2, ensure_ascii=False)

        # Force garbage collection
        import gc
        gc.collect()

    # Calculate final statistics
    total_pages = sum(r.get('total_pages', 0) for r in submission['results'] if 'total_pages' in r)
    total_detections = sum(r.get('summary', {}).get('total_detections', 0) for r in submission['results'] if 'summary' in r)

    submission['metadata']['total_pages'] = total_pages
    submission['metadata']['total_detections'] = total_detections

    # Final save
    final_output = Path('submission.json')
    with open(final_output, 'w', encoding='utf-8') as f:
        json.dump(submission, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 50)
    print("✅ DONE!")
    print(f"📁 Output: {final_output}")
    print(f"📊 Processed: {len(submission['results'])} PDFs")
    print(f"📄 Total pages: {total_pages}")
    print(f"🔍 Total detections: {total_detections}")

    file_size_mb = final_output.stat().st_size / (1024 * 1024)
    print(f"💾 File size: {file_size_mb:.2f} MB")

if __name__ == '__main__':
    main()
