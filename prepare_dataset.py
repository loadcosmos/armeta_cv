"""
Dataset Preparation for Kaggle - SIMPLIFIED VERSION
Uses single selected_annotations.json file

Usage:
    python prepare_dataset_kaggle.py

Expected structure:
    /kaggle/input/armeta-docs/data/
    ├── pdf/                                      (45 PDF files)
    └── annotations/
        └── selected_annotations.json             (single JSON with all annotations)

Output:
    /kaggle/working/data/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
    └── data.yaml
"""

import json
import cv2
import numpy as np
from pathlib import Path
from pdf2image import convert_from_path
from tqdm import tqdm
import yaml
from sklearn.model_selection import train_test_split
import shutil

# ============= CONFIGURATION =============
CLASS_MAPPING = {
    'signature': 0,
    'stamp': 1,
    'qr': 2,
    'qr_code': 2,  # alias
    'seal': 1,      # alias for stamp
}

# Kaggle paths
INPUT_DIR = Path('/kaggle/input/armeta-docs/data')
OUTPUT_DIR = Path('/kaggle/working/data')

PDF_DIR = INPUT_DIR / 'pdf'
ANNOTATIONS_FILE = INPUT_DIR / 'annotations' / 'selected_annotations.json'

DPI = 200
TRAIN_SPLIT = 0.8

# ============= FUNCTIONS =============
def load_annotations(json_path):
    """
    Load selected_annotations.json

    Expected format:
    {
      "pdf_name.pdf": {
        "page_1": {
          "annotations": [
            {"annotation_123": {"category": "signature", "bbox": {...}}},
            ...
          ]
        },
        "page_2": {...}
      }
    }
    """
    print(f"📖 Loading annotations from {json_path}...")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    annotations_by_image = {}

    # Parse the nested structure: PDF → pages → annotations
    if isinstance(data, dict):
        for pdf_filename, pages_data in data.items():
            # Remove .pdf extension for matching
            pdf_basename = pdf_filename.replace('.pdf', '')

            if isinstance(pages_data, dict):
                # Iterate through pages (page_1, page_2, etc.)
                for page_key, page_data in pages_data.items():
                    if not isinstance(page_data, dict):
                        continue

                    # Extract page number from "page_X"
                    if page_key.startswith('page_'):
                        page_num = page_key.split('_')[1]
                    else:
                        continue

                    # Get annotations for this page
                    annotations_list = page_data.get('annotations', [])

                    if not annotations_list:
                        continue

                    # Create image name that matches PDF conversion: basename_page_N.png
                    image_name = f"{pdf_basename}_page_{page_num}"

                    # Unwrap nested annotation structure
                    # Each annotation is: {"annotation_XXX": {category, bbox, ...}}
                    unwrapped_annotations = []
                    for ann_wrapper in annotations_list:
                        if isinstance(ann_wrapper, dict):
                            # Get the actual annotation (first value in dict)
                            for ann_id, ann_data in ann_wrapper.items():
                                if isinstance(ann_data, dict):
                                    unwrapped_annotations.append(ann_data)
                                    break

                    # Store annotations AND page_size for proper scaling
                    page_size = page_data.get('page_size', {})
                    annotations_by_image[image_name] = {
                        'annotations': unwrapped_annotations,
                        'page_size': page_size
                    }

    print(f"✅ Loaded annotations for {len(annotations_by_image)} images")

    # Debug: show first few entries
    if len(annotations_by_image) > 0:
        print(f"   Sample images: {list(annotations_by_image.keys())[:3]}")

    return annotations_by_image

def convert_pdf_to_images(pdf_path, output_dir, dpi=200):
    """Convert PDF to images"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = convert_from_path(str(pdf_path), dpi=dpi)

    saved_paths = []
    pdf_name = Path(pdf_path).stem

    for i, img in enumerate(images):
        img_path = output_dir / f"{pdf_name}_page_{i+1}.png"
        img.save(img_path, 'PNG')
        saved_paths.append(img_path)

    return saved_paths

def parse_annotation(ann):
    """Parse single annotation to extract class and bbox"""
    # Extract class
    class_name = None
    for key in ['label', 'category', 'class', 'name', 'category_name']:
        if key in ann:
            class_name = str(ann[key]).lower()
            break

    if not class_name:
        return None, None

    # Map to class ID
    class_id = None
    for key, value in CLASS_MAPPING.items():
        if key in class_name:
            class_id = value
            break

    if class_id is None:
        return None, None

    # Extract bbox
    bbox = None

    # Format 1: bbox as dict {"x": ..., "y": ..., "width": ..., "height": ...}
    if 'bbox' in ann:
        bbox_data = ann['bbox']
        if isinstance(bbox_data, dict) and all(k in bbox_data for k in ['x', 'y', 'width', 'height']):
            x, y, w, h = bbox_data['x'], bbox_data['y'], bbox_data['width'], bbox_data['height']
        # Format 2: bbox as array [x, y, width, height]
        elif isinstance(bbox_data, (list, tuple)) and len(bbox_data) == 4:
            x, y, w, h = bbox_data
        else:
            return None, None

    # Format 3: points (polygon)
    elif 'points' in ann:
        points = ann['points']
        xs = [p[0] if isinstance(p, (list, tuple)) else p['x'] for p in points]
        ys = [p[1] if isinstance(p, (list, tuple)) else p['y'] for p in points]
        x, y = min(xs), min(ys)
        w, h = max(xs) - x, max(ys) - y

    # Format 4: x, y, width, height as separate fields (root level)
    elif all(k in ann for k in ['x', 'y', 'width', 'height']):
        x, y = ann['x'], ann['y']
        w, h = ann['width'], ann['height']

    else:
        return None, None

    # Ensure all values are valid numbers
    try:
        x, y, w, h = float(x), float(y), float(w), float(h)
    except (ValueError, TypeError):
        return None, None

    return class_id, (x, y, w, h)

def convert_to_yolo_format(annotations, img_width, img_height, json_page_size=None):
    """
    Convert annotations to YOLO format

    Args:
        annotations: List of annotation dicts
        img_width: Actual image width (after PDF conversion)
        img_height: Actual image height (after PDF conversion)
        json_page_size: Original page size from JSON annotations (dict with 'width', 'height')
    """
    yolo_lines = []

    # Calculate scaling factors if JSON page size is different from actual image size
    scale_x = 1.0
    scale_y = 1.0

    if json_page_size and 'width' in json_page_size and 'height' in json_page_size:
        json_width = json_page_size['width']
        json_height = json_page_size['height']

        if json_width > 0 and json_height > 0:
            scale_x = img_width / json_width
            scale_y = img_height / json_height

    for ann in annotations:
        class_id, bbox = parse_annotation(ann)

        if class_id is None or bbox is None:
            continue

        x, y, w, h = bbox

        # Scale coordinates from JSON size to actual image size
        x = x * scale_x
        y = y * scale_y
        w = w * scale_x
        h = h * scale_y

        # Convert to YOLO format (normalized)
        x_center = (x + w / 2) / img_width
        y_center = (y + h / 2) / img_height
        norm_width = w / img_width
        norm_height = h / img_height

        # Validate
        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and
                0 < norm_width <= 1 and 0 < norm_height <= 1):
            continue

        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}")

    return yolo_lines

def prepare_dataset():
    """Main pipeline"""
    print("=" * 70)
    print("🚀 DATASET PREPARATION FOR KAGGLE")
    print("=" * 70)

    # Check paths
    if not PDF_DIR.exists():
        raise FileNotFoundError(f"PDF directory not found: {PDF_DIR}")

    if not ANNOTATIONS_FILE.exists():
        raise FileNotFoundError(f"Annotations file not found: {ANNOTATIONS_FILE}")

    # Load annotations
    annotations_by_file = load_annotations(ANNOTATIONS_FILE)

    # Create temp directories
    images_temp = OUTPUT_DIR / 'images_temp'
    labels_temp = OUTPUT_DIR / 'labels_temp'
    images_temp.mkdir(parents=True, exist_ok=True)
    labels_temp.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("📄 STEP 1: Converting PDFs to images...")
    print("=" * 70)

    pdf_files = sorted(PDF_DIR.glob('*.pdf'))
    print(f"Found {len(pdf_files)} PDF files")

    all_image_paths = []

    for pdf_path in tqdm(pdf_files, desc="Converting PDFs"):
        img_paths = convert_pdf_to_images(pdf_path, images_temp, dpi=DPI)
        all_image_paths.extend(img_paths)

    print(f"✅ Created {len(all_image_paths)} images")

    print("\n" + "=" * 70)
    print("📝 STEP 2: Creating YOLO labels...")
    print("=" * 70)

    total_annotations = 0
    images_with_labels = []

    for img_path in tqdm(all_image_paths, desc="Processing images"):
        # Image name without extension: "pdf_name_page_3"
        img_stem = img_path.stem

        # Try to find matching annotations data (contains 'annotations' and 'page_size')
        ann_data = annotations_by_file.get(img_stem)

        if ann_data is None:
            continue

        # Extract annotations and page_size
        if isinstance(ann_data, dict):
            matching_anns = ann_data.get('annotations', [])
            page_size = ann_data.get('page_size', {})
        else:
            # Fallback for old format (if any)
            matching_anns = ann_data
            page_size = {}

        if len(matching_anns) == 0:
            continue

        # Get image dimensions
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]

        # Convert to YOLO format with scaling
        yolo_lines = convert_to_yolo_format(matching_anns, w, h, json_page_size=page_size)

        if len(yolo_lines) == 0:
            continue

        # Save label file
        label_path = labels_temp / f"{img_path.stem}.txt"
        with open(label_path, 'w') as f:
            f.write('\n'.join(yolo_lines))

        total_annotations += len(yolo_lines)
        images_with_labels.append((img_path, label_path))

    print(f"✅ Created {len(images_with_labels)} label files with {total_annotations} annotations")

    if len(images_with_labels) == 0:
        raise ValueError("No images with labels found! Check annotation file format.")

    print("\n" + "=" * 70)
    print("🔀 STEP 3: Creating train/val split...")
    print("=" * 70)

    train_pairs, val_pairs = train_test_split(
        images_with_labels,
        train_size=TRAIN_SPLIT,
        random_state=42,
        shuffle=True
    )

    print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}")

    # Create final structure
    for split, pairs in [('train', train_pairs), ('val', val_pairs)]:
        split_img_dir = OUTPUT_DIR / split / 'images'
        split_lbl_dir = OUTPUT_DIR / split / 'labels'
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path, lbl_path in pairs:
            shutil.copy(img_path, split_img_dir / img_path.name)
            shutil.copy(lbl_path, split_lbl_dir / lbl_path.name)

    # Clean up temp
    shutil.rmtree(images_temp)
    shutil.rmtree(labels_temp)

    print("\n" + "=" * 70)
    print("📋 STEP 4: Creating data.yaml...")
    print("=" * 70)

    data_yaml = {
        'path': str(OUTPUT_DIR.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'nc': 3,
        'names': {0: 'signature', 1: 'stamp', 2: 'qr'}
    }

    yaml_path = OUTPUT_DIR / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print(f"✅ Created {yaml_path}")

    print("\n" + "=" * 70)
    print("✨ DATASET READY!")
    print("=" * 70)
    print(f"📁 Location: {OUTPUT_DIR}")
    print(f"📊 Train: {len(train_pairs)} images")
    print(f"📊 Val: {len(val_pairs)} images")
    print(f"📝 Annotations: {total_annotations}")
    print(f"\n🚀 Next: python train_yolov8s_optimized.py")

if __name__ == '__main__':
    prepare_dataset()
