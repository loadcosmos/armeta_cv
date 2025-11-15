# 🚀 Kaggle Quick Start - 3 Steps

## ⏱️ Total Time: ~1 hour

---

## 📦 Step 1: Upload Data to Kaggle (5 min)

### 1.1 Create Kaggle Dataset

1. Go to https://www.kaggle.com/datasets
2. Click "New Dataset"
3. Upload folder structure:
   ```
   data/
   ├── pdf/                                (45 PDF files)
   └── annotations/
       └── selected_annotations.json       (ONLY THIS FILE!)
   ```
4. Dataset name: **`armeta-docs`**
5. Click "Create"

**IMPORTANT:**
- Upload the ENTIRE `data/` folder with this structure
- Use ONLY `selected_annotations.json`, NOT `masked_annotations.json`!

---

## 🔧 Step 2: Create Kaggle Notebook (2 min)

1. Go to https://www.kaggle.com/code
2. Click "New Notebook"
3. Settings (top right):
   - ✅ **Accelerator: GPU T4 x2**
   - ✅ **Internet: ON**
4. Add Data:
   - Click "Add data" → Search "armeta-docs" → Add

---

## 💻 Step 3: Run Code (50 min)

Copy-paste each cell and run:

### CELL 1: Setup (2 min)

```python
# Install dependencies
!pip install -q ultralytics opencv-python-headless pdf2image scikit-learn

# Clone repository
!git clone https://github.com/loadcosmos/armeta_cv.git
%cd armeta_cv

# Check GPU
import torch
print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
```

### CELL 2: Prepare Dataset (15 min)

```python
# Convert PDF + JSON → YOLO dataset
!python prepare_dataset.py

# Verify
!echo "Train images:" && ls /kaggle/working/data/train/images | wc -l
!echo "Val images:" && ls /kaggle/working/data/val/images | wc -l
```

**Expected output:**
```
Train images: 103
Val images: 26
```

### CELL 3: Train Model (40 min)

```python
# Train YOLOv8s
!python train_yolov8s_optimized.py
```

**Expected results:**
- mAP50: ~0.61
- Stamp: 0.99+
- Signature: 0.55-0.60
- QR: 0.25-0.30

### CELL 4: Validate (2 min)

```python
from ultralytics import YOLO

# Load best model
model = YOLO('/kaggle/working/runs/detect/train/weights/best.pt')

# Validate
results = model.val(data='/kaggle/working/data/data.yaml')

# Print metrics
print(f"\n📊 Final Metrics:")
print(f"mAP50: {results.box.map50:.3f}")
print(f"Precision: {results.box.mp:.3f}")
print(f"Recall: {results.box.mr:.3f}")
```

### CELL 5: Test Hybrid QR Detection (5 min)

```python
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

# Hybrid detection function
def detect_opencv_qr(image):
    qr_detector = cv2.QRCodeDetector()
    retval, decoded_info, points, _ = qr_detector.detectAndDecodeMulti(image)
    return points if retval else []

# Test on validation image
val_images = sorted(Path('/kaggle/working/data/val/images').glob('*.png'))
test_img_path = val_images[0]

# Load image
img = cv2.imread(str(test_img_path))

# YOLO detection
yolo_results = model.predict(img, conf=0.25, imgsz=1024, verbose=False)[0]
yolo_qr_count = sum(1 for box in yolo_results.boxes if int(box.cls) == 2)

# OpenCV detection
opencv_qr_points = detect_opencv_qr(img)
opencv_qr_count = len(opencv_qr_points) if opencv_qr_points is not None else 0

print(f"🔷 YOLO found: {yolo_qr_count} QR codes")
print(f"🟢 OpenCV found: {opencv_qr_count} QR codes")
print(f"✅ Hybrid total: {max(yolo_qr_count, opencv_qr_count)} QR codes")
```

### CELL 6: Save Results (1 min)

```python
import shutil
from pathlib import Path

# Create output dir
output_dir = Path('/kaggle/working/output')
output_dir.mkdir(exist_ok=True)

# Copy model
shutil.copy(
    '/kaggle/working/runs/detect/train/weights/best.pt',
    output_dir / 'best.pt'
)

# Copy training results
shutil.copy(
    '/kaggle/working/runs/detect/train/results.csv',
    output_dir / 'training_results.csv'
)

print("✅ Results saved to /kaggle/working/output/")
print("Go to Output tab (right panel) to download")
```

---

## 📥 Step 4: Download Results

1. Stop notebook (to save outputs)
2. Click **"Output"** tab (right panel)
3. Download:
   - `best.pt` (your trained model)
   - `training_results.csv` (metrics)

---

## ✅ Success Checklist

- [ ] Dataset uploaded to Kaggle
- [ ] Notebook created (GPU + Internet ON)
- [ ] Data added to notebook
- [ ] All cells executed without errors
- [ ] Training completed (~40 min)
- [ ] mAP50 ≥ 0.60
- [ ] best.pt downloaded

---

## 🐛 Troubleshooting

### Error: "PDF directory not found"

Check dataset name and structure matches:
```python
# In prepare_dataset.py line 44:
INPUT_DIR = Path('/kaggle/input/armeta-docs/data')  # Must match your dataset name + /data

# Verify structure:
!ls -la /kaggle/input/armeta-docs/data/
# Should see: pdf/ and annotations/ folders
```

### Error: "CUDA out of memory"

Edit `train_yolov8s_optimized.py` line 30:
```python
'batch': 2  # Reduce from 4 to 2
```

### Error: "No images with labels found"

Check JSON format:
```python
# Debug annotations
import json
with open('/kaggle/input/armeta-docs/data/annotations/selected_annotations.json') as f:
    data = json.load(f)
    print(type(data))
    print(list(data.keys())[:5] if isinstance(data, dict) else data[:2])
```

---

## 🎯 What's Next?

After training completes:

1. **Add Killer Features** (multi-page PDF, validation, batch processing)
2. **Create Presentation** (7-10 slides)
3. **Test Streamlit app locally**
4. **Prepare for submission**

---

## ⏱️ Timeline

| Step | Time | Status |
|------|------|--------|
| Upload data | 5 min | Manual |
| Create notebook | 2 min | Manual |
| Setup | 2 min | Auto |
| Prepare dataset | 15 min | Auto |
| Train model | 40 min | Auto |
| Validate | 2 min | Auto |
| Test hybrid | 5 min | Auto |
| Save results | 1 min | Auto |
| **TOTAL** | **~1 hour** | |

---

**Ready? Start here: https://www.kaggle.com/datasets** 🚀
