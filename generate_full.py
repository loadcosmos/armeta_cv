#!/usr/bin/env python3
"""
Process remaining 45 PDFs from data/pdf/
Safe for low memory - processes one at a time
"""

import json
from pathlib import Path
from datetime import datetime
import sys

from enhanced_inference import EnhancedDocumentProcessor

def main():
    print("🔧 FULL DATASET GENERATOR (45 PDFs from data/pdf/)")
    print("=" * 60)

    # Find PDFs in data/pdf/
    pdf_dir = Path('data/pdf')
    pdfs = sorted([p for p in pdf_dir.glob('*.pdf') if 'Zone.Identifier' not in str(p)])

    print(f"\n✓ Found {len(pdfs)} PDF files in data/pdf/")

    # Load progress if exists
    output_file = Path('submission_full_progress.json')
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
                'source': 'data/pdf/',
                'total_pdfs': len(pdfs),
                'model': 'best.pt',
                'conf_threshold': 0.25,
                'iou_threshold': 0.45,
                'dpi': 150
            },
            'results': []
        }
        processed_files = set()

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

    # Process PDFs one by one
    success_count = 0
    error_count = 0

    for i, pdf_path in enumerate(pdfs, 1):
        pdf_name = pdf_path.name

        # Skip if already processed
        if pdf_name in processed_files:
            print(f"[{i}/{len(pdfs)}] ⏭️  Skipping {pdf_name} (already done)")
            continue

        # Show file size
        size_mb = pdf_path.stat().st_size / (1024 * 1024)
        print(f"[{i}/{len(pdfs)}] 📄 {pdf_name} ({size_mb:.2f} MB)")

        try:
            # Process with minimal options
            results = processor.process_pdf(
                pdf_path=str(pdf_path),
                dpi=150,  # Low DPI for memory
                output_dir=None,
                generate_html=False,
                save_visualizations=False
            )

            # Add to results
            results['relative_path'] = str(pdf_path.relative_to('data'))
            submission['results'].append(results)

            # Show summary
            pages = results.get('total_pages', 0)
            detections = results.get('summary', {}).get('total_detections', 0)
            print(f"   ✅ Success - {pages} pages, {detections} detections")
            success_count += 1

        except Exception as e:
            print(f"   ⚠️  Error: {str(e)[:100]}")
            # Add error entry
            submission['results'].append({
                'pdf': pdf_name,
                'relative_path': str(pdf_path.relative_to('data')),
                'error': str(e),
                'status': 'failed'
            })
            error_count += 1

        # Save progress immediately after each file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(submission, f, indent=2, ensure_ascii=False)

        # Force cleanup
        import gc
        gc.collect()

        # Show progress
        processed = success_count + error_count
        print(f"   Progress: {processed}/{len(pdfs)} | Success: {success_count} | Errors: {error_count}\n")

    # Calculate statistics
    total_pages = sum(r.get('total_pages', 0) for r in submission['results'] if 'total_pages' in r)
    total_detections = sum(r.get('summary', {}).get('total_detections', 0) for r in submission['results'] if 'summary' in r)

    submission['metadata']['total_pages'] = total_pages
    submission['metadata']['total_detections'] = total_detections
    submission['metadata']['success_count'] = success_count
    submission['metadata']['error_count'] = error_count

    # Save final
    final_output = Path('submission_full.json')
    with open(final_output, 'w', encoding='utf-8') as f:
        json.dump(submission, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("✅ PROCESSING COMPLETE!")
    print("=" * 60)
    print(f"\n📁 Output: {final_output}")
    print(f"📊 Processed: {len(submission['results'])} PDFs")
    print(f"   ✅ Success: {success_count}")
    print(f"   ⚠️  Errors: {error_count}")
    print(f"\n📄 Total pages: {total_pages}")
    print(f"🔍 Total detections: {total_detections}")

    file_size_mb = final_output.stat().st_size / (1024 * 1024)
    print(f"💾 File size: {file_size_mb:.2f} MB")

if __name__ == '__main__':
    main()
