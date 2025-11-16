# Setup Guide - Armeta CV Document Detection

## Quick Start (5 minutes)

### 1. Virtual Environment Setup

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install system dependencies
# Ubuntu/Debian:
sudo apt-get update
sudo apt-get install -y poppler-utils libgl1 libzbar0

# macOS:
brew install poppler zbar

# Verify QR decoder works
python -c "from pyzbar import pyzbar; print('✓ QR decoder ready!')"
```

### 2. Get Your Trained Model

```bash
# Download best.pt from your Kaggle training
# Place it in the project root or models/ folder
# The system will auto-detect it in these locations:
#   - best.pt
#   - models/best.pt
#   - runs/detect/train2/weights/best.pt
#   - weights/best.pt
```

### 3. Test the System

**Option A: Enhanced Inference (Command Line)**
```bash
# Test single document with all features
python enhanced_inference.py \
  --input data/test/ТЗ-2.pdf \
  --output demo/ \
  --html \
  --validator contract

# Open demo/ТЗ-2/report.html in browser to see results
```

**Option B: Mobile App (Web Interface)**
```bash
# Run locally
streamlit run mobile_app.py

# For mobile access (camera requires HTTPS):
# 1. Install ngrok: https://ngrok.com/download
# 2. Run: ngrok http 8501
# 3. Open the https://... URL on your phone
```

### 4. Batch Processing

```bash
# Process all test PDFs
python enhanced_inference.py \
  --input data/test/ \
  --output production_results/ \
  --batch \
  --html

# Check batch_summary.json for statistics
```

## Model Performance

Your trained model (mAP50: 88.1%):
- **QR codes**: 99.5% mAP50, 100% recall ⭐
- **Stamps**: 87.0% mAP50, 91.7% recall
- **Signatures**: 77.6% mAP50, 68.4% recall

## Killer Features

1. **QR Code Decoding** - Extracts actual data (URLs, emails, text)
2. **Document Validation** - Business rules for contracts/licenses
3. **HTML Reports** - Professional, client-ready output
4. **Mobile App** - Camera capture + multi-page scanning

## Troubleshooting

### QR Decoder Not Working
```bash
# Ubuntu/Debian
sudo apt-get install libzbar0

# macOS
brew install zbar

# Verify
python -c "from pyzbar import pyzbar; print('OK')"
```

### PDF Conversion Fails
```bash
# Ubuntu/Debian
sudo apt-get install poppler-utils

# macOS
brew install poppler

# Verify
python -c "from pdf2image import convert_from_path; print('OK')"
```

### Camera Not Working in Mobile App
- Camera API requires HTTPS
- Use ngrok for local testing
- Or deploy to Streamlit Cloud (free HTTPS)

### Model Not Found
```bash
# Make sure best.pt is in one of these locations:
ls best.pt
ls models/best.pt
ls runs/detect/train2/weights/best.pt

# Or specify manually:
python enhanced_inference.py --model /path/to/best.pt --input document.pdf
```

## File Structure

```
armeta_cv/
├── best.pt                      # Your trained model (download from Kaggle)
├── enhanced_inference.py        # Complete pipeline (CLI)
├── mobile_app.py               # Mobile web interface
├── inference.py                # Base YOLO detector
├── qr_decoder.py               # QR content extraction
├── validator.py                # Document validation
├── html_reporter.py            # HTML report generation
├── requirements.txt            # Python dependencies
├── data/
│   └── test/                   # Test PDFs
├── demo/                       # Test output
└── production_results/         # Batch output
```

## Next Steps

1. **Test locally**: Run enhanced_inference.py with test PDFs
2. **Test mobile app**: Run streamlit app and try camera
3. **Prepare presentation**: Show HTML reports and live demo
4. **Deploy for judges**: Use ngrok or Streamlit Cloud

## Deployment Options

See `DEPLOYMENT.md` for detailed deployment instructions including:
- Local network access
- ngrok tunnel (recommended for demo)
- Streamlit Cloud
- Docker deployment
- Security considerations
