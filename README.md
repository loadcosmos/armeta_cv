# 🚀 Digital Inspector - AI Document Detection System

**Production-ready document detection system with QR decoding, validation, and mobile support**

This project provides a complete solution for detecting signatures, stamps, and QR codes in documents using YOLOv8s object detection combined with QR decoding and document validation capabilities.

## 🎯 Features

- **Object Detection**: Detect signatures, stamps, and QR codes in documents
- **QR Code Decoding**: Extract and classify QR code content (URLs, emails, phones, etc.)
- **Document Validation**: Validate documents based on business rules
- **HTML Reports**: Generate beautiful, professional reports
- **Mobile Support**: Camera integration for mobile document scanning
- **Batch Processing**: Process multiple documents at once

## 📊 Performance Metrics

### Model Performance (YOLOv8s - 11.2M parameters)

| Metric | Overall | QR Code | Stamp | Signature |
|--------|---------|-------|-----------|
| **mAP50** | **88.1%** | **99.5%** ⭐ | 87.0% | 77.6% |
| **Recall** | 86.7% | **100%** ⭐ | 91.7% | 68.4% |
| **Precision** | 95.1% | 98.9% | 97.2% | 89.3% |

## 🚀 Quick Start

### Prerequisites

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install system dependencies
# Ubuntu/Debian:
sudo apt-get install poppler-utils libgl1 libzbar0

# macOS:
brew install poppler zbar
```

### Usage

#### Single File Inference
```bash
python app.py --mode inference --source path/to/document.pdf --model best.pt
```

#### Training
```bash
python app.py --mode train
```

#### Web Application
```bash
python app.py --mode app
```

#### Complete Workflow
```bash
python app.py --mode all
```

## 🏗️ Architecture

The system is organized as a single comprehensive application:

- `app.py` - Main application containing all functionality
- `best.pt` - Trained YOLO model weights
- `data/` - Input data directory
- `requirements.txt` - Python dependencies

## 📁 Project Structure

```
├── app.py                 # Main application (all-in-one)
├── best.pt               # Trained model weights
├── requirements.txt      # Python dependencies
├── data/                 # Input data
│   ├── pdf/             # PDF documents
│   ├── annotations/     # Annotation files
│   └── test/            # Test documents
└── README.md            # This file
```

## 💡 Key Components

### Document Detection
- YOLOv8s model for object detection
- Custom training for signatures, stamps, and QR codes
- High accuracy for small objects using 1024px image size

### QR Code Processing
- Decodes QR content using pyzbar and OpenCV
- Classifies data types (URLs, emails, phones, etc.)
- Quality assessment for decoded content

### Document Validation
- Business rule engine for document validation
- Predefined validators for contracts and licenses
- Custom validation rules support

### HTML Reporting
- Professional, responsive HTML reports
- Visual detection previews
- QR data tables and validation status
- Self-contained reports with embedded images

## 🌐 Deployment Options

### Local Network
```bash
python app.py --mode app
# Access from mobile: http://YOUR_IP:8501
```

### Streamlit Cloud
Deploy at https://streamlit.io/cloud for free HTTPS access.

## 📈 Comparison with Basic Detection

| Feature | Basic YOLO | Our System |
|---------|-----------|------------|
| Object Detection | ✓ | ✓ |
| QR Code Detection | ✓ | ✓ |
| **QR Code Decoding** | ✗ | ✓ |
| **Data Type Classification** | ✗ | ✓ |
| **Document Validation** | ✗ | ✓ |
| **Business Rules** | ✗ | ✓ |
| **HTML Reports** | ✗ | ✓ |
| **Mobile App** | ✗ | ✓ |
| **Camera Support** | ✗ | ✓ |

## 🚀 Future Enhancements

- REST API for integration
- OCR for text extraction
- Barcode support (Code128, Code39)
- Multi-language support
- Cloud storage integration
- Advanced overlapping detection
- Model quantization (ONNX, TensorRT)

## 🙏 Acknowledgments

- **Ultralytics** - YOLOv8 framework
- **OpenCV** - Image processing
- **pyzbar** - QR code decoding
- **Streamlit** - Web framework

## 📄 License

MIT License - See LICENSE file
