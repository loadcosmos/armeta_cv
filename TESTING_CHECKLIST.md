# Testing Checklist - Before Presentation

Complete these tests to ensure everything works for your hackathon demo.

## ✅ Pre-Testing Setup

### 1. Download Trained Model
```bash
# Download best.pt from your Kaggle training notebook
# Place it in the project root directory

# Verify it exists:
ls -lh best.pt
# Should show ~22.6 MB file
```

### 2. Install Dependencies
```bash
# Activate virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install Python packages
pip install -r requirements.txt

# Install system dependencies
# Ubuntu/Debian:
sudo apt-get update
sudo apt-get install -y poppler-utils libgl1 libzbar0

# macOS:
brew install poppler zbar

# Verify QR decoder works:
python -c "from pyzbar import pyzbar; print('✓ QR decoder ready!')"
```

---

## 🧪 Test 1: Enhanced Inference (Command Line)

### Single Document Test
```bash
# Test with one PDF from your test set
python enhanced_inference.py \
  --input data/test/ТЗ-2.pdf \
  --output test_results/ \
  --html \
  --validator contract

# Expected output:
# - test_results/ТЗ-2/results.json
# - test_results/ТЗ-2/report.html
# - test_results/ТЗ-2/page_*.jpg
```

**What to check:**
- [ ] Script runs without errors
- [ ] JSON file contains detections with QR data
- [ ] HTML report opens in browser
- [ ] Validation status shows (Valid/Warning/Invalid)
- [ ] Visualizations show bounding boxes and labels
- [ ] QR codes are decoded (check "data" field in JSON)

### Batch Processing Test
```bash
# Test with all PDFs
python enhanced_inference.py \
  --input data/test/ \
  --output batch_results/ \
  --batch \
  --html

# Expected output:
# - batch_results/batch_summary.json
# - batch_results/[pdf_name]/... (for each PDF)
```

**What to check:**
- [ ] All PDFs processed successfully
- [ ] batch_summary.json shows statistics
- [ ] Each PDF has its own subfolder with results
- [ ] HTML reports generated for each document

### Test Different Validators
```bash
# Test contract validator
python enhanced_inference.py \
  --input data/test/ТЗ-2.pdf \
  --validator contract \
  --html

# Test license validator
python enhanced_inference.py \
  --input data/test/ТЗ-2.pdf \
  --validator license \
  --html

# Test default validator
python enhanced_inference.py \
  --input data/test/ТЗ-2.pdf \
  --validator default \
  --html
```

**What to check:**
- [ ] Different validation rules applied
- [ ] Error messages make sense
- [ ] Contract requires 2+ signatures
- [ ] License requires stamp + signature + readable QR

### Test Overlapping Detection
```bash
# Test with lower thresholds for overlapping objects
python enhanced_inference.py \
  --input data/test/ТЗ-2.pdf \
  --conf 0.20 \
  --iou 0.3 \
  --html
```

**What to check:**
- [ ] More detections found (overlapping signatures under stamps)
- [ ] No excessive false positives

---

## 📱 Test 2: Mobile App (Web Interface)

### Local Testing
```bash
# Run the app
streamlit run mobile_app.py

# Open in browser: http://localhost:8501
```

**What to check:**
- [ ] App loads without errors
- [ ] Three modes available: Camera, Upload, Batch
- [ ] Upload mode works with local files
- [ ] Detections displayed correctly
- [ ] QR data shown in colored boxes
- [ ] Validation results displayed
- [ ] Download buttons work (JSON, HTML)

### Camera Testing (Desktop)
```bash
# Run with HTTPS (required for camera)
# Option 1: ngrok
ngrok http 8501

# Option 2: Streamlit Cloud (see DEPLOYMENT.md)
```

**What to check:**
- [ ] Camera permission requested
- [ ] Camera preview shows
- [ ] Take photo button works
- [ ] Photo processed and results shown

### Multi-Page Scanning Test
```bash
# In mobile app:
# 1. Select "Multi-Page Document" mode
# 2. Take/upload first page
# 3. Click "Add to Document"
# 4. Take/upload second page
# 5. Click "Add to Document"
# 6. Click "Create PDF & Process"
```

**What to check:**
- [ ] Multiple pages can be added
- [ ] Page counter updates
- [ ] Preview shows all pages
- [ ] PDF created successfully
- [ ] Combined results shown
- [ ] Download works

### Mobile Device Testing
```bash
# Use ngrok to get HTTPS URL
ngrok http 8501

# Open the https://... URL on your phone
```

**What to check:**
- [ ] App loads on mobile browser
- [ ] Camera button appears
- [ ] Camera opens native phone camera
- [ ] Photo captured successfully
- [ ] Results display properly on small screen
- [ ] Responsive design works

---

## 🔍 Test 3: Feature Verification

### QR Code Decoding
**Expected behavior:**
- Detects QR code bounding box
- Decodes actual content
- Classifies data type (URL, email, phone, text)
- Shows quality assessment (high/medium/low)

**Test with:**
- Documents containing QR codes
- Check JSON output for `qr_data` field

### Document Validation
**Expected behavior:**
- Counts signatures, stamps, QR codes
- Applies business rules
- Returns errors or warnings
- Shows validation status

**Test with:**
- Document with 2+ signatures (should be valid for contract)
- Document with 1 signature (should have error for contract)
- Document with unreadable QR (should have warning)

### HTML Reports
**Expected behavior:**
- Modern, responsive design
- Embedded images (no external files)
- Color-coded validation status
- Tables with detection data
- QR data highlighted

**Test with:**
- Open report.html in different browsers
- Check on mobile browser
- Verify images load correctly

---

## 🚨 Common Issues & Fixes

### Issue: "ModuleNotFoundError: No module named 'pyzbar'"
**Fix:**
```bash
# Install system dependency
sudo apt-get install libzbar0  # Ubuntu/Debian
brew install zbar              # macOS

# Reinstall Python package
pip install pyzbar
```

### Issue: "PDF conversion failed"
**Fix:**
```bash
# Install poppler
sudo apt-get install poppler-utils  # Ubuntu/Debian
brew install poppler                # macOS
```

### Issue: "Model not found"
**Fix:**
```bash
# Make sure best.pt is in one of these locations:
# - ./best.pt
# - ./models/best.pt
# - ./runs/detect/train2/weights/best.pt

# Or specify manually:
python enhanced_inference.py --model /path/to/best.pt --input file.pdf
```

### Issue: "Camera doesn't work"
**Fix:**
- Camera API requires HTTPS
- Use ngrok for local testing
- Or deploy to Streamlit Cloud

### Issue: "No QR codes decoded"
**Fix:**
- Check if QR codes are in the image
- Verify libzbar0 installed
- Try adjusting image quality/size
- Check QR code quality in PDF

---

## 📊 Performance Benchmarks

Record these metrics for your presentation:

### Detection Speed
```bash
# Run with timing
time python enhanced_inference.py \
  --input data/test/ТЗ-2.pdf \
  --output timing_test/ \
  --html
```

**Record:**
- [ ] Total processing time
- [ ] Time per page
- [ ] Time for QR decoding
- [ ] Time for HTML generation

### Accuracy
**Check from training results:**
- [ ] Overall mAP50: 88.1%
- [ ] QR mAP50: 99.5%
- [ ] Stamp mAP50: 87.0%
- [ ] Signature mAP50: 77.6%

### QR Decode Rate
```bash
# Process all test PDFs and count:
# - Total QR codes detected
# - Total QR codes successfully decoded
# - Decode success rate = decoded/detected * 100%
```

**Target:** >95% decode success rate

---

## ✅ Final Checklist

Before the presentation, verify:

### Code
- [ ] All dependencies installed
- [ ] best.pt model file present
- [ ] No errors when running scripts
- [ ] Git repository up to date

### Features
- [ ] QR decoding works
- [ ] Document validation works
- [ ] HTML reports generated
- [ ] Mobile app runs
- [ ] Camera capture works (with HTTPS)
- [ ] Multi-page scanning works

### Documentation
- [ ] README.md updated with new features
- [ ] SETUP.md provides clear instructions
- [ ] KILLER_FEATURES.md shows advantages
- [ ] DEPLOYMENT.md explains deployment

### Demo Preparation
- [ ] Sample PDFs ready
- [ ] Mobile demo URL ready (ngrok or Streamlit Cloud)
- [ ] HTML report examples saved
- [ ] Performance metrics noted
- [ ] Presentation slides prepared
- [ ] Story ready: problem → solution → results

---

## 🎯 Demo Script (Recommended Flow)

### 1. Problem Statement (30 seconds)
"Document validation is manual, slow, and error-prone. QR codes need to be scanned separately."

### 2. Our Solution (30 seconds)
"We built an AI system that detects signatures, stamps, and QR codes - AND reads the QR content - AND validates business rules - all automatically."

### 3. Live Demo (2-3 minutes)

**Option A: Command Line (Fast)**
```bash
python enhanced_inference.py \
  --input sample.pdf \
  --validator contract \
  --html
```
Show the HTML report.

**Option B: Mobile App (Impressive)**
1. Open mobile app on phone
2. Take photo of document
3. Show real-time detection
4. Show QR data decoded
5. Show validation result

### 4. Key Metrics (30 seconds)
- 88.1% mAP50 overall accuracy
- 99.5% mAP50 for QR codes (nearly perfect!)
- 100% recall on QR codes
- 2-3 seconds per page processing

### 5. Killer Features (1 minute)
- QR decoding (not just detection!)
- Document validation (business rules)
- HTML reports (client-ready)
- Mobile app (camera + multi-page)

### 6. Technical Excellence (30 seconds)
- Fixed coordinate scaling bug (497x improvement!)
- Clean architecture
- Production-ready code
- Extensive documentation

**Total: ~5 minutes**

---

## 📝 Notes

- **Priority**: Test enhanced_inference.py first (core functionality)
- **Second priority**: Test mobile app locally
- **If time permits**: Test mobile app on phone with ngrok
- **Save examples**: Keep good HTML reports to show in presentation
- **Backup plan**: Have screenshots ready if live demo fails

Good luck! 🚀
