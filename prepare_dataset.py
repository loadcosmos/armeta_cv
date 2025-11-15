"""
Complete Dataset Preparation Script for Kaggle
Converts PDF + JSON annotations → YOLO format dataset

Steps:
1. Convert PDFs to images
2. Convert JSON annotations to YOLO format
3. Create train/val split
4. Generate data.yaml

Run in Kaggle:
!python prepare_dataset.py --pdf-dir /kaggle/input/your-dataset/pdf --annotations-dir /kaggle/input/your-dataset/annotations
"""

import json
import cv2
import numpy as np
from pathlib import Path
from pdf2image import convert_from_path
from tqdm import tqdm
import yaml
import argparse
from sklearn.model_selection import train_test_split
import shutil

# ============= CONFIGURATION =============
CLASS_MAPPING = {
    'signature': 0,
    'stamp': 1,
    'qr': 2
}

# ============= FUNCTIONS =============
def convert_pdf_to_images(pdf_path, output_dir, dpi=200):
    """
    Convert PDF to images (one per page)

    Args:
        pdf_path: Path to PDF file
        output_dir: Directory to save images
        dpi: Resolution (higher = better quality but slower)

    Returns:
        List of saved image paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert PDF to images
    images = convert_from_path(pdf_path, dpi=dpi)

    saved_paths = []
    pdf_name = Path(pdf_path).stem

    for i, img in enumerate(images):
        # Save as PNG
        img_path = output_dir / f"{pdf_name}_page_{i+1}.png"
        img.save(img_path, 'PNG')
        saved_paths.append(img_path)

    return saved_paths

def convert_json_to_yolo(json_path, image_width, image_height, output_path):
    """
    Convert JSON annotation to YOLO format

    YOLO format: <class_id> <x_center> <y_center> <width> <height>
    All values normalized to [0, 1]

    Args:
        json_path: Path to JSON annotation file
        image_width: Image width in pixels
        image_height: Image height in pixels
        output_path: Path to save YOLO .txt file
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    yolo_lines = []

    # Handle different JSON formats
    annotations = []

    # Format 1: COCO-like format
    if 'annotations' in data:
        annotations = data['annotations']
    # Format 2: Direct list
    elif isinstance(data, list):
        annotations = data
    # Format 3: Custom format with 'objects' or 'shapes'
    elif 'objects' in data:
        annotations = data['objects']
    elif 'shapes' in data:
        annotations = data['shapes']

    for ann in annotations:
        # Extract class name
        class_name = None
        if 'label' in ann:
            class_name = ann['label'].lower()
        elif 'category' in ann:
            class_name = ann['category'].lower()
        elif 'class' in ann:
            class_name = ann['class'].lower()
        elif 'name' in ann:
            class_name = ann['name'].lower()

        # Map to class ID
        class_id = None
        for key, value in CLASS_MAPPING.items():
            if class_name and key in class_name:
                class_id = value
                break

        if class_id is None:
            continue  # Skip unknown classes

        # Extract bounding box
        bbox = None

        # Format 1: bbox as [x, y, width, height]
        if 'bbox' in ann:
            bbox = ann['bbox']
            x, y, w, h = bbox
        # Format 2: points (polygon) - compute bbox
        elif 'points' in ann:
            points = ann['points']
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            x, y = min(xs), min(ys)
            w, h = max(xs) - x, max(ys) - y
        # Format 3: x, y, width, height as separate fields
        elif 'x' in ann and 'width' in ann:
            x, y = ann['x'], ann['y']
            w, h = ann['width'], ann['height']
        else:
            continue  # Skip if no bbox info

        # Convert to YOLO format (normalized x_center, y_center, width, height)
        x_center = (x + w / 2) / image_width
        y_center = (y + h / 2) / image_height
        norm_width = w / image_width
        norm_height = h / image_height

        # Validate coordinates (must be in [0, 1])
        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and
                0 < norm_width <= 1 and 0 < norm_height <= 1):
            print(f"Warning: Invalid bbox in {json_path}: {ann}")
            continue

        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}")

    # Write YOLO format file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(yolo_lines))

    return len(yolo_lines)

def prepare_yolo_dataset(pdf_dir, annotations_dir, output_dir, train_split=0.8, dpi=200):
    """
    Complete pipeline: PDF + JSON → YOLO dataset

    Args:
        pdf_dir: Directory with PDF files
        annotations_dir: Directory with JSON annotation files
        output_dir: Output directory for YOLO dataset
        train_split: Train/val split ratio
        dpi: PDF conversion DPI
    """
    pdf_dir = Path(pdf_dir)
    annotations_dir = Path(annotations_dir)
    output_dir = Path(output_dir)

    # Create output structure
    images_dir = output_dir / 'images_temp'
    images_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("📄 STEP 1: Converting PDFs to images...")
    print("=" * 60)

    # Convert all PDFs
    pdf_files = sorted(pdf_dir.glob('*.pdf'))
    all_image_paths = []

    for pdf_path in tqdm(pdf_files, desc="Converting PDFs"):
        img_paths = convert_pdf_to_images(pdf_path, images_dir, dpi=dpi)
        all_image_paths.extend(img_paths)

    print(f"✅ Converted {len(pdf_files)} PDFs → {len(all_image_paths)} images")

    print("\n" + "=" * 60)
    print("📝 STEP 2: Converting JSON annotations to YOLO format...")
    print("=" * 60)

    # Convert annotations
    labels_dir = output_dir / 'labels_temp'
    labels_dir.mkdir(parents=True, exist_ok=True)

    annotation_files = sorted(annotations_dir.glob('*.json'))
    total_annotations = 0

    for json_path in tqdm(annotation_files, desc="Converting annotations"):
        # Find corresponding image
        json_name = json_path.stem

        # Try to match image by name
        matching_images = list(images_dir.glob(f"{json_name}*.png"))

        if not matching_images:
            # Try without page suffix
            base_name = json_name.replace('_page_1', '').replace('_page_2', '')
            matching_images = list(images_dir.glob(f"{base_name}*.png"))

        for img_path in matching_images:
            # Get image dimensions
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            h, w = img.shape[:2]

            # Convert to YOLO format
            label_path = labels_dir / f"{img_path.stem}.txt"
            num_annotations = convert_json_to_yolo(json_path, w, h, label_path)
            total_annotations += num_annotations

    print(f"✅ Created {len(list(labels_dir.glob('*.txt')))} label files with {total_annotations} annotations")

    print("\n" + "=" * 60)
    print("🔀 STEP 3: Creating train/val split...")
    print("=" * 60)

    # Get all images with labels
    label_files = sorted(labels_dir.glob('*.txt'))
    image_label_pairs = []

    for label_path in label_files:
        img_path = images_dir / f"{label_path.stem}.png"
        if img_path.exists():
            image_label_pairs.append((img_path, label_path))

    # Split
    train_pairs, val_pairs = train_test_split(
        image_label_pairs,
        train_size=train_split,
        random_state=42,
        shuffle=True
    )

    print(f"📊 Train: {len(train_pairs)} images, Val: {len(val_pairs)} images")

    # Create final directory structure
    for split, pairs in [('train', train_pairs), ('val', val_pairs)]:
        split_img_dir = output_dir / split / 'images'
        split_lbl_dir = output_dir / split / 'labels'
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path, lbl_path in pairs:
            shutil.copy(img_path, split_img_dir / img_path.name)
            shutil.copy(lbl_path, split_lbl_dir / lbl_path.name)

    # Clean up temp directories
    shutil.rmtree(images_dir)
    shutil.rmtree(labels_dir)

    print("\n" + "=" * 60)
    print("📋 STEP 4: Creating data.yaml...")
    print("=" * 60)

    # Create data.yaml
    data_yaml = {
        'path': str(output_dir.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'nc': 3,
        'names': {0: 'signature', 1: 'stamp', 2: 'qr'}
    }

    yaml_path = output_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print(f"✅ Created {yaml_path}")

    print("\n" + "=" * 60)
    print("✨ DATASET READY!")
    print("=" * 60)
    print(f"📁 Location: {output_dir}")
    print(f"📊 Train images: {len(train_pairs)}")
    print(f"📊 Val images: {len(val_pairs)}")
    print(f"📝 Total annotations: {total_annotations}")
    print(f"\n🚀 Next step: python train_yolov8s_optimized.py")

# ============= MAIN =============
def main():
    parser = argparse.ArgumentParser(description='Prepare YOLO dataset from PDF + JSON')
    parser.add_argument('--pdf-dir', type=str, required=True,
                       help='Directory with PDF files')
    parser.add_argument('--annotations-dir', type=str, required=True,
                       help='Directory with JSON annotations')
    parser.add_argument('--output-dir', type=str, default='./data',
                       help='Output directory for YOLO dataset')
    parser.add_argument('--train-split', type=float, default=0.8,
                       help='Train/val split ratio')
    parser.add_argument('--dpi', type=int, default=200,
                       help='PDF to image DPI (higher = better quality)')

    args = parser.parse_args()

    prepare_yolo_dataset(
        pdf_dir=args.pdf_dir,
        annotations_dir=args.annotations_dir,
        output_dir=args.output_dir,
        train_split=args.train_split,
        dpi=args.dpi
    )

if __name__ == '__main__':
    main()
