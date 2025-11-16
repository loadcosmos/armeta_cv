# ⚡ Quick Start - 5 Minutes to Testing

**Everything you need to start testing immediately.**

---

## Step 1: Get the Model (1 min)

```bash
# Download best.pt from your Kaggle training notebook
# Place it in the project root directory
# Verify:
ls -lh best.pt
# Should show ~22.6 MB
```

---

## Step 2: Install (2 min)

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install everything
pip install -r requirements.txt

# System dependencies
# Ubuntu/Debian:
sudo apt-get install -y poppler-utils libgl1 libzbar0

# macOS:
brew install poppler zbar

# Verify QR decoder:
python -c "from pyzbar import pyzbar; print('✓ Ready!')"
```

---

## Step 3: Test Enhanced Inference (1 min)

```bash
python enhanced_inference.py \
  --input data/test/ТЗ-2.pdf \
  --output demo/ \
  --html \
  --validator contract
```

**Check output:**
- `demo/ТЗ-2/report.html` ← Open this in browser!
- `demo/ТЗ-2/results.json` ← Full data
- `demo/ТЗ-2/page_*.jpg` ← Visualizations

---

## Step 4: Test Mobile App (1 min)

```bash
streamlit run mobile_app.py
```

**Open:** http://localhost:8501

**Try:**
- Upload a PDF
- See detections
- Check QR data
- See validation results

---

## 🎯 For Mobile Demo (Optional, 2 min)

```bash
# Terminal 1: Run app
streamlit run mobile_app.py

# Terminal 2: Create tunnel
ngrok http 8501
```

**Open the https://... URL on your phone**
- Camera will work (requires HTTPS)
- Try multi-page scanning

---

## 📊 Quick Commands Reference

### Single Document
```bash
python enhanced_inference.py --input doc.pdf --html
```

### Batch Processing
```bash
python enhanced_inference.py --input pdfs/ --batch --html
```

### Contract Validation
```bash
python enhanced_inference.py --input doc.pdf --validator contract --html
```

### License Validation
```bash
python enhanced_inference.py --input doc.pdf --validator license --html
```

### Overlapping Objects
```bash
python enhanced_inference.py --input doc.pdf --conf 0.20 --iou 0.3 --html
```

---

## 🚨 Quick Fixes

### "No module named 'pyzbar'"
```bash
sudo apt-get install libzbar0  # Ubuntu
brew install zbar              # macOS
pip install pyzbar
```

### "PDF conversion failed"
```bash
sudo apt-get install poppler-utils  # Ubuntu
brew install poppler                # macOS
```

### "Model not found"
```bash
# Make sure best.pt is in project root
ls best.pt
# Or specify path:
python enhanced_inference.py --model /path/to/best.pt --input doc.pdf
```

---

## 🎤 5-Minute Demo Script

### 1. Problem (30s)
"Document validation is manual and slow. QR codes need separate scanning."

### 2. Solution (30s)
"AI system that detects signatures, stamps, QR codes AND reads QR content AND validates business rules."

### 3. Demo (2m)
```bash
# Run this:
python enhanced_inference.py \
  --input data/test/ТЗ-2.pdf \
  --validator contract \
  --html

# Open: demo/ТЗ-2/report.html
```

**Show:**
- Detections on page
- QR codes decoded (URLs/data)
- Validation result (Valid/Invalid)
- Professional formatting

### 4. Metrics (30s)
- 88.1% mAP50 overall
- 99.5% mAP50 QR codes
- 100% recall QR codes
- 2-3 seconds per page

### 5. Killer Features (1m)
1. QR decoding (read content!)
2. Validation (business rules)
3. HTML reports (client-ready)
4. Mobile app (camera + multi-page)

### 6. Why We Win (30s)
"Complete production system. Not just detection - understanding, validation, presentation."

---

## 📁 File Structure (Clean!)

```
armeta_cv/
├── enhanced_inference.py    # 🌟 Main CLI tool
├── mobile_app.py           # 📱 Mobile app
├── inference.py            # Base detector
├── qr_decoder.py           # QR reader
├── validator.py            # Validation
├── html_reporter.py        # Reports
├── best.pt                 # Model (download!)
├── requirements.txt        # Dependencies
└── data/test/              # Test PDFs
```

---

## 🎯 What Makes You Win

✅ **88.1% mAP50** - High accuracy
✅ **99.5% QR mAP50** - Nearly perfect
✅ **QR Decoding** - Read content, not just detect
✅ **Validation** - Business rules engine
✅ **HTML Reports** - Professional output
✅ **Mobile App** - Camera + multi-page
✅ **Clean Code** - Production-ready
✅ **Great Docs** - 7 comprehensive guides

**Fixed critical bug: 0.2% → 99.5% mAP50 (497x improvement!)**

---

## 📋 Pre-Demo Checklist

- [ ] best.pt downloaded
- [ ] Dependencies installed
- [ ] Test command works
- [ ] HTML report opens
- [ ] Mobile app runs
- [ ] Demo script practiced
- [ ] Metrics memorized (88.1%, 99.5%, 100%)

---

## 🎓 Key Points for Presentation

**Technical:** Fixed coordinate scaling bug → 497x improvement

**Innovation:** First to combine YOLO + QR decoding + validation

**Business:** Solves real compliance problems, client-ready reports

**Demo:** Live mobile app with camera OR command-line with HTML

**Metrics:** 88.1% mAP50, 99.5% QR, 100% recall

---

## 📞 Need Help?

- **Setup**: See SETUP.md
- **Testing**: See TESTING_CHECKLIST.md
- **Features**: See KILLER_FEATURES.md
- **Deploy**: See DEPLOYMENT.md
- **Overview**: See HACKATHON_SUMMARY.md

---

**⚡ You're ready! Good luck! 🚀**

**Performance: 88.1% mAP50 | QR: 99.5% | Innovation: 100%**
