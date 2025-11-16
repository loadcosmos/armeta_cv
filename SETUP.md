# ⚡ Setup & Deployment Guide

Complete guide to install, test, and deploy the document detection system.

---

## 📦 Quick Setup (5 minutes)

### Step 1: Environment Setup

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install Python dependencies
pip3 install -r requirements.txt
```

### Step 2: System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y poppler-utils libgl1 libzbar0
```

**macOS:**
```bash
brew install poppler zbar
```

**Windows:**
- Download poppler: https://github.com/oschwartz10612/poppler-windows/releases
- Download zbar: http://zbar.sourceforge.net/download.html
- Add to PATH

**Verify installation:**
```bash
python -c "from pyzbar import pyzbar; print('✓ QR decoder ready!')"
python -c "from pdf2image import convert_from_path; print('✓ PDF converter ready!')"
```

### Step 3: Get Model Weights

Download `best.pt` (21.5 MB) from your Kaggle training and place in project root:

```bash
# Verify model
ls -lh best.pt
# Should show ~21.5 MB

# The system will auto-detect it in these locations:
#   - best.pt (root)
#   - models/best.pt
#   - runs/detect/train2/weights/best.pt
#   - weights/best.pt
```

---

## ✅ Testing

### Test 1: Single Document (CLI)

```bash
python enhanced_inference.py \
  --input data/test/ТЗ-2.pdf \
  --output demo/ \
  --html \
  --validator contract
```

**Check output:**
- `demo/ТЗ-2/report.html` ← Open in browser!
- `demo/ТЗ-2/results.json` ← Full detection data
- `demo/ТЗ-2/page_*.jpg` ← Visualizations

### Test 2: Batch Processing

```bash
python enhanced_inference.py \
  --input data/test/ \
  --output results/ \
  --batch \
  --html
```

**Check:**
- `results/*/report.html` ← HTML report for each PDF
- `results/batch_summary.json` ← Aggregate statistics

### Test 3: Web Application

```bash
streamlit run mobile_app.py
```

**Open:** http://localhost:8501

**Try:**
1. Upload a PDF from `data/test/`
2. See detection results with colored boxes
3. Check QR code data (if present)
4. Review validation status

---

## 🌐 Deployment Options

### Option 1: Local Network (LAN Demo)

**For presenting to judges on same WiFi:**

```bash
# Run app accessible from network
streamlit run mobile_app.py --server.port 8501 --server.address 0.0.0.0

# Find your IP
ip addr show  # Linux
ipconfig      # Windows
ifconfig      # macOS

# Share with judges
# They open: http://YOUR_IP:8501
```

### Option 2: ngrok Tunnel (Public Demo)

**For remote judges or mobile camera testing:**

```bash
# Terminal 1: Run app
streamlit run mobile_app.py

# Terminal 2: Create tunnel
# Download ngrok first: https://ngrok.com/download
ngrok http 8501

# Share the https://xxxx.ngrok.io URL
# Works anywhere, camera enabled (HTTPS)!
```

**Perfect for hackathon demos!**

### Option 3: Streamlit Cloud (Production)

**Free, automatic HTTPS, always online:**

1. Push code to GitHub (already done!)
2. Go to https://streamlit.io/cloud
3. Sign in with GitHub
4. Click "New app"
5. Select your repo
6. Main file: `mobile_app.py`
7. Click "Deploy"!

**Note about model:**
- GitHub has 100MB file limit
- Upload `best.pt` to GitHub LFS or Google Drive
- Add download logic in app, or use `git lfs`

### Option 4: Docker (Advanced)

**For cloud deployment (AWS, GCP, Azure):**

```dockerfile
# Dockerfile (already created)
FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    poppler-utils libgl1 libzbar0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "mobile_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0"]
```

```bash
# Build and run
docker build -t document-scanner .
docker run -p 8501:8501 document-scanner

# Access: http://localhost:8501
```

---

## 🔧 Configuration

### Streamlit Settings

Edit `.streamlit/config.toml`:

```toml
[server]
maxUploadSize = 10  # Max file size in MB
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#667eea"
backgroundColor = "#ffffff"
```

### Model Settings

Edit thresholds in your script:

```python
detector = DocumentDetector(
    conf_threshold=0.25,  # Lower = more detections (more false positives)
    iou_threshold=0.45    # Higher = fewer duplicate boxes
)
```

### Validation Rules

Create custom validators in `validator.py`:

```python
from validator import DocumentValidator, ValidationRule

class MyValidator(DocumentValidator):
    def _get_default_rules(self):
        return [
            ValidationRule(
                name="custom_rule",
                description="My custom validation",
                check_function=lambda r: r['counts'].get('stamp', 0) >= 3,
                severity='error'
            )
        ]
```

---

## 🐛 Troubleshooting

### QR Decoder Not Working

**Error:** `Unable to find zbar shared library`

```bash
# Ubuntu/Debian
sudo apt-get install libzbar0

# macOS
brew install zbar

# Windows
# Download from: http://zbar.sourceforge.net/download.html
# Add DLL to PATH or place in project root

# Verify
python -c "from pyzbar import pyzbar; print('OK')"
```

### PDF Conversion Fails

**Error:** `Unable to get page count. Is poppler installed?`

```bash
# Ubuntu/Debian
sudo apt-get install poppler-utils

# macOS
brew install poppler

# Windows
# Download from: https://github.com/oschwartz10612/poppler-windows/releases
# Extract and add bin/ to PATH

# Verify
pdftoppm -v
```

### Model Not Found

**Error:** `FileNotFoundError: best.pt`

```bash
# Check if model exists
ls -lh best.pt

# If not, download from Kaggle and place in root

# Or specify path manually
python enhanced_inference.py --model /path/to/best.pt --input doc.pdf
```

### Camera Not Working in Mobile App

**Issue:** Camera button doesn't appear or shows error

**Solution:**
- Camera API requires HTTPS
- Use ngrok for local testing (creates HTTPS tunnel)
- Or deploy to Streamlit Cloud (automatic HTTPS)
- HTTP (localhost) won't work on mobile browsers

### Out of Memory

**Issue:** App crashes with large PDFs

**Solutions:**
1. Reduce image size (DPI):
   ```python
   python enhanced_inference.py --input doc.pdf --dpi 150  # Default: 200
   ```

2. Increase upload limit in `.streamlit/config.toml`:
   ```toml
   [server]
   maxUploadSize = 50  # Increase from 10MB
   ```

3. Process pages sequentially instead of loading all

---

## 📊 Performance Tuning

### Speed Optimization

```bash
# Use GPU (if available)
# PyTorch will automatically detect CUDA

# Check GPU usage
nvidia-smi

# Reduce confidence threshold for faster processing
python enhanced_inference.py --input doc.pdf --conf 0.3  # Higher = faster

# Skip QR decoding for speed
# (Modify enhanced_inference.py to skip QR decoder)
```

### Accuracy vs Speed

| Setting | Speed | Accuracy | Use Case |
|---------|-------|----------|----------|
| `conf=0.15, iou=0.3` | Slow | High recall | Missing detections |
| `conf=0.25, iou=0.45` | Medium | Balanced | **Default** |
| `conf=0.40, iou=0.60` | Fast | High precision | Clean documents |

---

## 📁 Project Structure

```
armeta_cv/
├── best.pt                    # Model weights (21.5 MB)
├── requirements.txt           # Python dependencies
│
├── enhanced_inference.py      # Complete pipeline (CLI)
├── mobile_app.py             # Web interface
│
├── inference.py              # YOLO detection
├── qr_decoder.py             # QR content extraction
├── validator.py              # Document validation
├── html_reporter.py          # HTML generation
│
├── prepare_dataset.py        # Training data prep
├── train_yolov8s_optimized.py # Model training
│
├── .streamlit/
│   └── config.toml          # Streamlit config
├── data/
│   └── test/                # Test PDFs
└── README.md
```

---

## 🚀 Next Steps

### For Hackathon Demo:

1. ✅ **Test locally** - Verify all features work
2. ✅ **Prepare samples** - Have good/bad document examples
3. ✅ **Deploy** - Use ngrok or Streamlit Cloud
4. ✅ **Create presentation** - Show HTML reports + live demo

### For Production:

1. Add authentication (Streamlit auth or OAuth)
2. Set up logging and monitoring
3. Implement rate limiting
4. Add database for results storage
5. Create REST API endpoints
6. Set up CI/CD pipeline

---

## 💡 Tips for Judges Demo

1. **Have fallback PDFs ready** - Pre-processed results in case of internet issues
2. **Show HTML reports** - More impressive than JSON
3. **Demo camera on phone** - Live document scanning is wow factor
4. **Explain validation rules** - Business value > technical details
5. **Compare with/without QR decoding** - Show the value-add

---

## 🔗 Useful Links

- **Ultralytics Docs**: https://docs.ultralytics.com/
- **Streamlit Docs**: https://docs.streamlit.io/
- **pyzbar Docs**: https://github.com/NaturalHistoryMuseum/pyzbar
- **ngrok Setup**: https://ngrok.com/docs/getting-started

---

**Questions?** Check README.md for technical details or test with `python check_system.py`
