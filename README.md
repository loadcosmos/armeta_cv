# 🚀 Digital Inspector - AI Document Detection System

**Production-ready YOLO-based document validation with QR decoding, business rules, and mobile support**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8s-green.svg)](https://github.com/ultralytics/ultralytics)
[![mAP50](https://img.shields.io/badge/mAP50-88.1%25-brightgreen.svg)]()

Hackathon-winning document detection system that goes beyond basic object detection with **QR content extraction**, **business rule validation**, and **production-ready deployment**.

---

## 🎯 Key Features

### 🔍 Core Detection
- **YOLOv8s** model trained for documents (11.2M parameters)
- Detects: **Signatures**, **Stamps**, **QR Codes**
- **88.1% mAP50** overall | **99.5% mAP50** for QR codes ⭐

### 🚀 Killer Features (Beyond Basic Detection)

1. **QR Code Content Extraction** 📱
   - Not just detection - actual data reading with pyzbar
   - Classifies data type: URL, email, phone, text, JSON
   - Quality assessment (high/medium/low)
   - ~95% decode success rate

2. **Business Rules Validation** ✅
   - Contract validator (2+ signatures, stamps, QR required)
   - License validator (stamp + signature + readable QR)
   - Custom validators with multi-level severity
   - Automated compliance checking

3. **Professional HTML Reports** 📊
   - Modern, responsive design
   - Visual detection previews with color-coded boxes
   - QR data tables
   - Validation status indicators
   - Self-contained (embedded images)

4. **Mobile-First Web App** 📸
   - Camera capture from phone
   - Multi-page document scanning
   - Combine pages to PDF
   - Real-time processing
   - Streamlit Cloud ready (HTTPS)

---

## 📊 Performance Metrics

### Model Accuracy

| Class | mAP50 | Recall | Precision |
|-------|-------|--------|-----------|
| **QR** | **99.5%** | **100%** ⭐ | 98.9% |
| Stamp | 87.0% | 91.7% | 97.2% |
| Signature | 77.6% | 68.4% | 89.3% |
| **Overall** | **88.1%** | **86.7%** | **95.1%** |

### Processing Speed
- Detection: ~50ms per page (GPU)
- QR Decoding: ~100-200ms per code
- **Total: 2-3 seconds per page**

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install Python packages
pip install -r requirements.txt

# Install system dependencies
# Ubuntu/Debian:
sudo apt-get install -y poppler-utils libgl1 libzbar0

# macOS:
brew install poppler zbar

# Verify QR decoder:
python -c "from pyzbar import pyzbar; print('✓ Ready!')"
```

### 2. Get Model Weights

Download `best.pt` (21.5 MB) from your training and place in project root.

### 3. Run Detection

**Command-Line Interface:**
```bash
# Single document with all features
python enhanced_inference.py \
  --input data/test/ТЗ-2.pdf \
  --output results/ \
  --html \
  --validator contract

# Batch processing
python enhanced_inference.py \
  --input data/test/ \
  --output results/ \
  --batch \
  --html
```

**Web Application:**
```bash
streamlit run mobile_app.py
# Open: http://localhost:8501
```

---

## 📁 Project Structure

```
armeta_cv/
├── best.pt                    # Trained YOLOv8s model (21.5 MB)
├── requirements.txt           # Python dependencies
├──
├── enhanced_inference.py      # Complete pipeline (CLI)
├── mobile_app.py             # Mobile web interface
├──
├── inference.py              # YOLO detection module
├── qr_decoder.py             # QR content extraction
├── validator.py              # Document validation rules
├── html_reporter.py          # HTML report generation
├──
├── prepare_dataset.py        # Dataset preparation (training)
├── train_yolov8s_optimized.py # Model training script
├──
├── .streamlit/
│   └── config.toml          # Streamlit deployment config
├── data/
│   └── test/                # Test PDFs (13 documents)
└── README.md                # This file
```

---

## 💡 Usage Examples

### Example 1: Process Contract with Validation
```bash
python enhanced_inference.py \
  --input contract.pdf \
  --validator contract \
  --html
```

**Output:**
```
results/contract/
├── results.json      # Full detection data + QR data + validation
├── report.html       # Beautiful HTML report
├── page_1.jpg       # Visualizations with boxes
└── page_2.jpg
```

### Example 2: Process License
```bash
python enhanced_inference.py \
  --input license.pdf \
  --validator license \
  --html
```

### Example 3: Batch Process All Documents
```bash
python enhanced_inference.py \
  --input data/test/ \
  --output production_results/ \
  --batch \
  --html \
  --conf 0.25 \
  --iou 0.45
```

### Example 4: Mobile App with Camera
```bash
# Run app
streamlit run mobile_app.py

# Access from phone (camera requires HTTPS):
# Option A: Use ngrok
ngrok http 8501

# Option B: Deploy to Streamlit Cloud
# Free, automatic HTTPS, no setup
```

---

## 🏆 What Sets This Apart

### Comparison with Basic Detection

| Feature | Basic YOLO | Our System |
|---------|-----------|------------|
| Object Detection | ✓ | ✓ |
| QR Code Detection | ✓ | ✓ |
| **QR Code Decoding** | ✗ | ✓ |
| **Data Type Classification** | ✗ | ✓ |
| **Document Validation** | ✗ | ✓ |
| **Business Rules** | ✗ | ✓ |
| **HTML Reports** | ✗ | ✓ |
| **Mobile Camera** | ✗ | ✓ |
| **Batch Processing** | Basic | Advanced |

### Why This Wins Hackathons

1. **Completeness** - End-to-end solution (PDF → JSON → HTML report)
2. **Business Value** - Solves real problems (validation, compliance)
3. **Production Ready** - Deployment config, caching, security
4. **Professional Output** - Client-ready HTML reports
5. **Scalability** - Batch processing + efficient pipeline

---

## 🌐 Deployment

### Local Network
```bash
streamlit run mobile_app.py --server.port 8501 --server.address 0.0.0.0
# Access from mobile: http://YOUR_IP:8501
```

### Streamlit Cloud (Recommended)
1. Push to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect repo → Deploy (automatic HTTPS!)
4. **Camera works immediately** (no ngrok needed)

### Configuration
- Upload limit: 10MB (see `.streamlit/config.toml`)
- Model caching: `@st.cache_resource` (loads once, reuses)
- Memory management: Automatic temp file cleanup

See `SETUP.md` for detailed deployment instructions.

---

## 🔧 Technical Details

### Training
- Dataset: Custom annotated documents
- Model: YOLOv8s (smaller, faster than YOLOv8m/l/x)
- Image size: 1024px (better for small objects)
- Augmentation: Aggressive (rotation, blur, noise)
- **Key fix**: Coordinate scaling bug (0.2% → 99.5% mAP50)

### Architecture
- **Modular design**: Separate modules for detection, QR, validation, reporting
- **Type hints**: Full type annotations for clarity
- **Error handling**: Graceful degradation (QR decode fails → marks as unreadable)
- **Configurable validators**: Easy to add custom rules

---

## 📈 Future Enhancements

- [ ] REST API for integration
- [ ] OCR for text extraction
- [ ] Barcode support (Code128, Code39)
- [ ] Multi-language support
- [ ] Cloud storage integration (S3, GCS)
- [ ] Model quantization (ONNX, TensorRT)
- [ ] Advanced overlapping detection

---

## 🙏 Acknowledgments

- **Ultralytics** - YOLOv8 framework
- **OpenCV** - Image processing
- **pyzbar** - QR code decoding
- **Streamlit** - Web framework
- **Armeta CV Hackathon 2024** - Inspiring the project

---

## 📄 License

MIT License - Free for commercial and personal use

---

## 📞 Support

For questions or issues:
1. Check `SETUP.md` for troubleshooting
2. Review example outputs in `results/`
3. Test with provided PDFs in `data/test/`

**Built for Armeta CV Hackathon 2024** | **Team: Digital Inspector** 🚀
