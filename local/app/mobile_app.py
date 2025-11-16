"""
Mobile-Friendly Streamlit App with Camera Support

Features:
- Take photo from mobile camera
- Multi-page document scanning
- Create PDF from photos
- Real-time detection
- QR decoding
- Document validation
- Download results

Usage:
    streamlit run mobile_app.py --server.port 8501 --server.address 0.0.0.0
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import base64
from pathlib import Path
import tempfile
from datetime import datetime
import json

# Import our modules
import os
try:
    from kaggle.hybrid.inference import DocumentDetector, find_model
    from kaggle.hybrid.qr_decoder import QRDecoder, format_qr_data
    from kaggle.hybrid.validator import DocumentValidator, ContractValidator, format_validation_report
    from kaggle.hybrid.html_reporter import HTMLReporter
except ImportError:
    st.error("⚠️ Please install all requirements: pip install -r requirements.txt")
    st.stop()

# Cache model loading for better performance
@st.cache_resource
def load_models():
    """Load and cache ML models"""
    detector = DocumentDetector(conf_threshold=0.25, iou_threshold=0.45)
    qr_decoder = QRDecoder()
    validator = DocumentValidator()
    reporter = HTMLReporter()
    return detector, qr_decoder, validator, reporter

# Page config
st.set_page_config(
    page_title="📱 Document Scanner",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for mobile
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 15px;
        border-radius: 10px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    .upload-box {
        border: 3px dashed #667eea;
        border-radius: 15px;
        padding: 30px;
        text-align: center;
        background: #f9fafb;
        margin: 20px 0;
    }
    .detection-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .qr-data {
        background: #f0f9ff;
        border-left: 4px solid #3b82f6;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        font-family: monospace;
    }
    .valid {
        background: #d1fae5;
        border-left: 4px solid #10b981;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .warning {
        background: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .invalid {
        background: #fee2e2;
        border-left: 4px solid #ef4444;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'pages' not in st.session_state:
    st.session_state.pages = []

# Load models with caching (only once across all sessions)
try:
    detector, qr_decoder, validator, reporter = load_models()
    st.session_state.detector = detector
    st.session_state.qr_decoder = qr_decoder
    st.session_state.validator = validator
    st.session_state.reporter = reporter
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.info("💡 Make sure best.pt is in the correct location")
    st.stop()

def process_image(image):
    """Process single image and return results"""
    # Convert PIL to OpenCV
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Detect
    detections = st.session_state.detector.detect_image(img_cv)

    # Decode QR
    detections = st.session_state.qr_decoder.decode_all_qr_codes(img_cv, detections)

    # Visualize
    img_vis = st.session_state.detector.visualize_detections(img_cv, detections)
    img_vis = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)

    return detections, Image.fromarray(img_vis)

def images_to_pdf(images, output_path):
    """Convert multiple images to PDF"""
    if images:
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            resolution=200.0
        )

def get_image_download_link(img, filename="image.jpg"):
    """Generate download link for image"""
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:image/jpeg;base64,{img_str}" download="{filename}">📥 Download Image</a>'
    return href

# Header
st.title("📱 Mobile Document Scanner")
st.markdown("**Detect signatures, stamps & QR codes** • Decode QR data • Validate documents")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")

    mode = st.radio(
        "📸 Scanning Mode",
        ["Single Photo", "Multi-Page Document"],
        help="Single: One photo\nMulti-Page: Scan multiple pages into one PDF"
    )

    validator_type = st.selectbox(
        "📋 Document Type",
        ["General", "Contract", "License"],
        help="Validation rules for document type"
    )

    conf_threshold = st.slider(
        "🎯 Confidence Threshold",
        0.1, 0.9, 0.25, 0.05,
        help="Lower = more detections (may include false positives)"
    )

    iou_threshold = st.slider(
        "🔲 Overlap Threshold",
        0.1, 0.9, 0.45, 0.05,
        help="Lower = allow more overlapping detections"
    )

    st.session_state.detector.conf_threshold = conf_threshold
    st.session_state.detector.iou_threshold = iou_threshold

    # Update validator
    if validator_type == "Contract":
        st.session_state.validator = ContractValidator()
    elif validator_type == "License":
        from kaggle.hybrid.validator import LicenseValidator
        st.session_state.validator = LicenseValidator()
    else:
        st.session_state.validator = DocumentValidator()

    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.markdown("""
    **YOLOv8s**
    - mAP50: 88.1%
    - QR: 99.5% (100% recall) ⭐
    - Stamp: 87.0%
    - Signature: 77.6%
    """)

# Main content
tab1, tab2, tab3 = st.tabs(["📷 Camera", "📁 Upload", "📊 Batch Process"])

with tab1:
    st.markdown("### 📸 Take Photo")
    st.info("💡 **Mobile users:** This will open your camera!")

    # Camera input
    camera_photo = st.camera_input("Take a photo of the document")

    if camera_photo:
        image = Image.open(camera_photo)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### Original")
            st.image(image, use_column_width=True)

        with col2:
            st.markdown("#### Detection Result")
            with st.spinner("🔍 Analyzing..."):
                detections, img_vis = process_image(image)
                st.image(img_vis, use_column_width=True)

        # Display results
        if detections:
            st.markdown("### 📋 Detection Results")

            counts = {}
            qr_data_list = []

            for det in detections:
                cls = det['class']
                counts[cls] = counts.get(cls, 0) + 1

                if cls == 'qr' and 'qr_data' in det:
                    qr_data_list.append(det['qr_data'])

            # Summary
            cols = st.columns(len(counts))
            for i, (cls, count) in enumerate(counts.items()):
                with cols[i]:
                    st.metric(cls.capitalize(), count)

            # QR data
            if qr_data_list:
                st.markdown("#### 🔍 QR Code Data")
                for i, qr in enumerate(qr_data_list, 1):
                    with st.expander(f"QR Code #{i}"):
                        if qr.get('data'):
                            st.markdown(f'<div class="qr-data"><strong>Type:</strong> {qr.get("data_type", "unknown")}<br><strong>Data:</strong> {qr["data"]}<br><strong>Quality:</strong> {qr.get("quality", "unknown")}</div>', unsafe_allow_html=True)
                        else:
                            st.warning("❌ Unreadable QR code")

            # Add to multi-page if in that mode
            if mode == "Multi-Page Document":
                if st.button("➕ Add to Document"):
                    st.session_state.pages.append({
                        'image': image,
                        'detections': detections,
                        'visualized': img_vis
                    })
                    st.success(f"✅ Added! Total pages: {len(st.session_state.pages)}")
        else:
            st.warning("No detections found. Try adjusting settings in sidebar.")

with tab2:
    st.markdown("### 📁 Upload Image or PDF")

    uploaded_file = st.file_uploader(
        "Choose file",
        type=['jpg', 'jpeg', 'png', 'pdf'],
        help="Upload image or PDF document"
    )

    if uploaded_file:
        if uploaded_file.type == 'application/pdf':
            st.info("📄 PDF processing coming soon! Use camera/upload for images now.")
        else:
            image = Image.open(uploaded_file)

            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("#### Original")
                st.image(image, use_column_width=True)

            with col2:
                st.markdown("#### Detection Result")
                with st.spinner("🔍 Analyzing..."):
                    detections, img_vis = process_image(image)
                    st.image(img_vis, use_column_width=True)

            if detections:
                st.markdown("### 📋 Detection Results")

                # Results similar to camera tab
                counts = {}
                for det in detections:
                    cls = det['class']
                    counts[cls] = counts.get(cls, 0) + 1

                cols = st.columns(len(counts))
                for i, (cls, count) in enumerate(counts.items()):
                    with cols[i]:
                        st.metric(cls.capitalize(), count)

with tab3:
    st.markdown("### 📊 Multi-Page Document")

    if st.session_state.pages:
        st.success(f"📄 {len(st.session_state.pages)} pages scanned")

        # Show pages
        for i, page in enumerate(st.session_state.pages, 1):
            with st.expander(f"Page {i}"):
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.image(page['image'], caption=f"Original - Page {i}", use_column_width=True)
                with col2:
                    st.image(page['visualized'], caption=f"Detected - Page {i}", use_column_width=True)

                # Detections
                counts = {}
                for det in page['detections']:
                    cls = det['class']
                    counts[cls] = counts.get(cls, 0) + 1
                st.write(f"**Detections:** {counts}")

        # Actions
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🗑️ Clear All"):
                st.session_state.pages = []
                st.rerun()

        with col2:
            if st.button("✅ Validate Document"):
                # Create results structure
                results = {
                    'pdf': 'scanned_document.pdf',
                    'total_pages': len(st.session_state.pages),
                    'pages': []
                }

                for i, page in enumerate(st.session_state.pages, 1):
                    results['pages'].append({
                        'page': i,
                        'detections': page['detections'],
                        'summary': {}
                    })

                # Validate
                validation = st.session_state.validator.validate(results)

                # Display validation
                status = validation['status']
                if status == 'valid':
                    st.markdown(f'<div class="valid">✅ <strong>Document is VALID</strong></div>', unsafe_allow_html=True)
                elif status == 'warning':
                    st.markdown(f'<div class="warning">⚠️ <strong>Document has WARNINGS</strong></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="invalid">❌ <strong>Document is INVALID</strong></div>', unsafe_allow_html=True)

                if validation['errors']:
                    st.error("**Errors:**")
                    for err in validation['errors']:
                        st.write(f"- {err['description']}")

                if validation['warnings']:
                    st.warning("**Warnings:**")
                    for warn in validation['warnings']:
                        st.write(f"- {warn['description']}")

        with col3:
            if st.button("📥 Download PDF"):
                with st.spinner("Creating PDF..."):
                    # Create PDF from images
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                        tmp_path = tmp.name
                        images = [page['image'].convert('RGB') for page in st.session_state.pages]
                        images_to_pdf(images, tmp_path)

                        # Provide download
                        with open(tmp_path, 'rb') as f:
                            pdf_bytes = f.read()

                        # Clean up temporary file
                        try:
                            os.remove(tmp_path)
                        except:
                            pass

                        st.download_button(
                            label="📄 Download PDF",
                            data=pdf_bytes,
                            file_name=f"scanned_document_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf"
                        )
    else:
        st.info("📸 No pages scanned yet. Use Camera tab to start scanning!")
        st.markdown("""
        **Multi-Page Workflow:**
        1. Go to **Camera** tab
        2. Take photo of each page
        3. Click **Add to Document**
        4. Return here to validate and download PDF
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6b7280;'>
    <p>🚀 Powered by YOLOv8 + QR Decoder + Validator</p>
    <p>Made for Armeta CV Hackathon 2024</p>
</div>
""", unsafe_allow_html=True)
