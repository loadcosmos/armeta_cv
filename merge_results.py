#!/usr/bin/env python3
"""
Merge results from test (13 PDFs) and full (45 PDFs) into final submission.json
Total: 58 PDFs
"""

import json
from pathlib import Path
from datetime import datetime

def main():
    print("🔀 MERGING RESULTS")
    print("=" * 60)

    # Load test results (13 PDFs from data/test/)
    test_file = Path('submission.json')
    full_file = Path('submission_full.json')

    results_combined = []
    total_pages = 0
    total_detections = 0

    # Load test results
    if test_file.exists():
        print(f"\n📂 Loading {test_file}...")
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)

        test_results = test_data.get('results', [])
        results_combined.extend(test_results)

        test_pages = sum(r.get('total_pages', 0) for r in test_results if 'total_pages' in r)
        test_dets = sum(r.get('summary', {}).get('total_detections', 0) for r in test_results if 'summary' in r)

        print(f"   ✅ Test set: {len(test_results)} PDFs, {test_pages} pages, {test_dets} detections")

        total_pages += test_pages
        total_detections += test_dets
    else:
        print(f"   ⚠️  {test_file} not found - skipping test set")

    # Load full results
    if full_file.exists():
        print(f"\n📂 Loading {full_file}...")
        with open(full_file, 'r', encoding='utf-8') as f:
            full_data = json.load(f)

        full_results = full_data.get('results', [])
        results_combined.extend(full_results)

        full_pages = sum(r.get('total_pages', 0) for r in full_results if 'total_pages' in r)
        full_dets = sum(r.get('summary', {}).get('total_detections', 0) for r in full_results if 'summary' in r)

        print(f"   ✅ Full set: {len(full_results)} PDFs, {full_pages} pages, {full_dets} detections")

        total_pages += full_pages
        total_detections += full_dets
    else:
        print(f"   ⚠️  {full_file} not found - skipping full set")

    if not results_combined:
        print("\n❌ No results to merge!")
        return

    # Count by class
    class_counts = {'signature': 0, 'stamp': 0, 'qr': 0}
    for result in results_combined:
        if 'summary' in result and 'by_class' in result['summary']:
            for cls, count in result['summary']['by_class'].items():
                if cls in class_counts:
                    class_counts[cls] += count

    # Count validation statuses
    validation_counts = {'valid': 0, 'warning': 0, 'invalid': 0, 'not_validated': 0}
    for result in results_combined:
        if 'validation' in result:
            status = result['validation'].get('status', 'unknown')
            if status in validation_counts:
                validation_counts[status] += 1
        else:
            validation_counts['not_validated'] += 1

    # Create final submission
    submission_final = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'description': 'Complete hackathon submission - all 58 PDFs',
            'total_pdfs': len(results_combined),
            'total_pages': total_pages,
            'total_detections': total_detections,
            'model': 'best.pt',
            'conf_threshold': 0.25,
            'iou_threshold': 0.45,
            'dpi': 150,
            'sources': {
                'test': len([r for r in results_combined if 'test/' in r.get('relative_path', '')]),
                'pdf': len([r for r in results_combined if 'pdf/' in r.get('relative_path', '')])
            },
            'detections_by_class': class_counts,
            'validation_summary': validation_counts
        },
        'results': results_combined
    }

    # Save final
    output = Path('submission_final.json')
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(submission_final, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("✅ MERGE COMPLETE!")
    print("=" * 60)
    print(f"\n📁 Output: {output}")
    print(f"\n📊 FINAL STATISTICS:")
    print(f"   Total PDFs: {len(results_combined)}")
    print(f"   Total Pages: {total_pages}")
    print(f"   Total Detections: {total_detections}")

    print(f"\n📂 By Source:")
    print(f"   data/test/: {submission_final['metadata']['sources']['test']} PDFs")
    print(f"   data/pdf/:  {submission_final['metadata']['sources']['pdf']} PDFs")

    print(f"\n🎯 By Class:")
    for cls, count in class_counts.items():
        print(f"   {cls}: {count}")

    print(f"\n✅ Validation:")
    for status, count in validation_counts.items():
        print(f"   {status}: {count}")

    file_size_mb = output.stat().st_size / (1024 * 1024)
    print(f"\n💾 File size: {file_size_mb:.2f} MB")

    print("\n🚀 Ready for hackathon submission!")

if __name__ == '__main__':
    main()
