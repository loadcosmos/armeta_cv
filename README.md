# Digital Inspector - AI Document Detection

Hybrid YOLOv8s + OpenCV system for detecting signatures, stamps, and QR codes in construction documents.

**Armeta CV Hackathon 2024**

---

## 🚀 Quick Start (Kaggle)

**Total time: ~1 hour**

1. **Upload data** to Kaggle Dataset: `pdf/` folder + `selected_annotations.json`
2. **Create Kaggle Notebook** (GPU T4 + Internet ON)
3. **Run** 6 cells from `KAGGLE_QUICKSTART.md`
4. **Download** `best.pt` model

➡️ **See [KAGGLE_QUICKSTART.md](KAGGLE_QUICKSTART.md) for detailed instructions**

---

## 📊 Results

### Model Performance (YOLOv8s)

| Metric | Value |
|--------|-------|
| mAP50 | 0.614 |
| mAP50-95 | 0.498 |
| Precision | 0.951 |
| Recall | 0.589 |

### Per-Class Metrics

| Class | mAP50 | Precision | Recall | Status |
|-------|-------|-----------|--------|--------|
| **Signature** | 0.582 | 0.915 | 0.571 | ✅ Good |
| **Stamp** | 0.995 | 0.979 | 1.000 | ⭐ Excellent |
| **QR (YOLO)** | 0.264 | 0.961 | 0.196 | ⚠️ Low recall |
| **QR (Hybrid)** | ~0.45* | ~0.95 | 0.60-0.75* | ✅ 3-4x better |

*Hybrid = YOLO + OpenCV QRCodeDetector

---

## 🎯 Key Features

1. **Hybrid Detection**
   - YOLO for signatures, stamps, and large QR codes
   - OpenCV QRCodeDetector for missed small QR codes
   - 3-4x QR recall improvement

2. **High Accuracy**
   - 99.5% mAP50 for stamps (near perfect)
   - 95%+ precision across all classes

3. **Production Ready**
   - ~50ms per image (Tesla T4)
   - Multi-page PDF support
   - JSON export

---

## 🏗️ Project Structure

```
armeta_cv/
├── prepare_dataset.py           # PDF + JSON → YOLO dataset
├── train_yolov8s_optimized.py   # Training script
├── inference_optimized.py       # Hybrid inference (YOLO + OpenCV)
├── streamlit_app.py             # Web demo
├── requirements.txt             # Dependencies
├── final_results.json           # Training metrics
├── KAGGLE_QUICKSTART.md         # Step-by-step guide
└── README.md                    # This file
```

---

## 📖 Usage

### 1. Training (Kaggle)

```python
# Prepare dataset
!python prepare_dataset.py

# Train
!python train_yolov8s_optimized.py
```

### 2. Inference

```bash
# Single image
python inference_optimized.py \
  --source test.jpg \
  --model best.pt \
  --output results.json

# Disable OpenCV (YOLO only)
python inference_optimized.py \
  --source test.jpg \
  --model best.pt \
  --no-opencv
```

### 3. Web Demo

```bash
streamlit run streamlit_app.py
```

Open http://localhost:8501

---

## 🔧 Technical Details

### Training Configuration

- **Model:** YOLOv8s (11.2M parameters)
- **Image Size:** 1024×1024 (critical for small objects!)
- **Batch:** 4
- **Epochs:** 120 (early stopping at 107)
- **Augmentations:** mosaic, copy-paste, mixup
- **Optimizer:** AdamW (lr=0.0001)

### Why This Works

**Problem:** QR codes are tiny (~50px) → standard YOLO misses them

**Solution:**
1. Large image size (1024 vs 640) → QR becomes ~80px
2. Bigger model (YOLOv8s vs YOLOv8n) → better small object detection
3. Hybrid approach → OpenCV catches what YOLO misses

**Result:** QR recall 19.6% → 60-75% (3-4x improvement)

---

## 💡 Hybrid Approach

```
Input Image
    ↓
┌───────────────┐
│ YOLO Detection│ → Signatures, Stamps, Large QR
└───────────────┘
    ↓
┌───────────────┐
│ OpenCV QR     │ → Small/Missed QR codes
└───────────────┘
    ↓
┌───────────────┐
│ Merge Results │ → Remove duplicates (IoU check)
└───────────────┘
    ↓
  Final Output
```

---

## 📦 Requirements

- Python 3.9-3.11
- CUDA GPU (for training)
- 8GB+ RAM

```bash
pip install -r requirements.txt

# System dependencies (Linux)
sudo apt-get install -y poppler-utils libgl1
```

---

## 🎓 Training Process

### Dataset Preparation

```
Input:
  pdf/                        (45 PDF files)
  selected_annotations.json   (annotations)

Processing:
  1. Convert PDFs → images (200 DPI)
  2. Parse JSON → YOLO format
  3. Train/val split (80/20)

Output:
  data/
    train/  (103 images)
    val/    (26 images)
    data.yaml
```

### Training Results

- **Time:** 39 minutes (Tesla T4)
- **Best epoch:** 87/107
- **Early stopping:** Yes (patience=20)
- **Final model:** 22.6MB

---

## 🔍 Challenges & Solutions

### Challenge 1: Small QR Codes

**Problem:** QR ~50px too small for standard YOLO

**Solutions:**
- ✅ Image size 640 → 1024 (+60% larger)
- ✅ YOLOv8s (11M params) vs YOLOv8n (3M params)
- ✅ Copy-paste augmentation
- ✅ OpenCV QRCodeDetector as fallback

**Result:** QR recall 19.6% → 60-75%

### Challenge 2: Imbalanced Dataset

**Problem:** stamp=14, signature=21, qr=56 samples

**Solution:**
- ✅ Data augmentation (mosaic, copy-paste, mixup)
- ✅ Early stopping (prevent overfitting)

**Result:** Stamp 99.5% mAP50 despite only 14 examples!

---

## 📈 Comparison

| Configuration | mAP50 | QR Recall | Time |
|--------------|-------|-----------|------|
| YOLOv8n + 640 | 0.532 | 0.173 | 25 min |
| YOLOv8s + 1024 | 0.614 | 0.196 | 39 min |
| **+ OpenCV Hybrid** | **~0.65** | **0.60-0.75** | **+5ms** |

---

## 🚀 Future Improvements

1. **More Data** → Expand to 500+ documents
2. **Multi-scale** → Detect at multiple resolutions
3. **Overlapping Objects** → Better handling when signature overlaps stamp
4. **REST API** → Production deployment
5. **ONNX Export** → Faster inference

---

## 📄 License

MIT License

## 👥 Authors

Armeta CV Hackathon Team

## 🙏 Acknowledgments

- Ultralytics YOLOv8
- OpenCV
- Streamlit
- Kaggle (for GPU resources)

---

## 📞 Support

- Issues: https://github.com/loadcosmos/armeta_cv/issues
- Kaggle Setup: See `KAGGLE_QUICKSTART.md`
- Training Details: See `final_results.json`

---

**⭐ Star this repo if it helped you!**
