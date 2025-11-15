"""
Streamlit Demo App - Digital Inspector
Armeta CV Hackathon - Document Detection

Run: streamlit run streamlit_app.py
"""

import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from pdf2image import convert_from_bytes
import io

# ============= CONFIG =============
MODEL_PATH = 'runs/detect/train/weights/best.pt'  # Update path
CLASS_NAMES = {0: 'signature', 1: 'stamp', 2: 'qr'}
CLASS_COLORS = {
    'signature': '#FF6B6B',  # Red
    'stamp': '#4ECDC4',      # Cyan
    'qr': '#95E1D3'          # Green
}

# ============= FUNCTIONS =============
@st.cache_resource
def load_model():
    """Load YOLO model (cached)"""
    return YOLO(MODEL_PATH)

def draw_detections(image, detections):
    """Draw bounding boxes on image"""
    draw = ImageDraw.Draw(image)

    # Try to load font (fallback to default if not available)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()

    for det in detections:
        cls = det['class']
        conf = det['confidence']
        box = det['bbox']

        # Draw rectangle
        color = CLASS_COLORS.get(cls, '#FFFFFF')
        draw.rectangle(
            [(box['x1'], box['y1']), (box['x2'], box['y2'])],
            outline=color,
            width=4
        )

        # Draw label
        label = f"{cls} {conf:.2f}"
        bbox = draw.textbbox((box['x1'], box['y1'] - 25), label, font=font)
        draw.rectangle(bbox, fill=color)
        draw.text((box['x1'], box['y1'] - 25), label, fill='white', font=font)

    return image

def predict_image(model, image, conf_threshold):
    """Run inference on image"""
    # Convert PIL to numpy
    img_array = np.array(image)

    # Predict
    results = model.predict(
        source=img_array,
        conf=conf_threshold,
        imgsz=1024,
        augment=True,
        device='cpu',  # Use 'cuda' if GPU available
        verbose=False
    )

    # Extract detections
    detections = []
    for r in results:
        boxes = r.boxes
        for i in range(len(boxes)):
            det = {
                'class': CLASS_NAMES[int(boxes.cls[i])],
                'class_id': int(boxes.cls[i]),
                'confidence': float(boxes.conf[i]),
                'bbox': {
                    'x1': float(boxes.xyxy[i][0]),
                    'y1': float(boxes.xyxy[i][1]),
                    'x2': float(boxes.xyxy[i][2]),
                    'y2': float(boxes.xyxy[i][3]),
                }
            }
            detections.append(det)

    return detections

# ============= STREAMLIT APP =============
def main():
    st.set_page_config(
        page_title="Digital Inspector",
        page_icon="🔍",
        layout="wide"
    )

    # Header
    st.title("🔍 Digital Inspector")
    st.markdown("**AI-powered detection of signatures, stamps, and QR codes in construction documents**")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        # Confidence threshold
        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.25,
            step=0.05,
            help="Lower = more detections (but more false positives)"
        )

        st.markdown("---")
        st.header("📊 Model Info")
        st.markdown(f"""
        - **Model:** YOLOv8s
        - **Classes:** signature, stamp, qr
        - **Input size:** 1024x1024
        - **TTA:** Enabled
        """)

        st.markdown("---")
        st.markdown("### 🏆 Armeta Hackathon 2024")
        st.markdown("Built with ❤️ using YOLOv8 & Streamlit")

    # Load model
    try:
        model = load_model()
        st.sidebar.success("✅ Model loaded")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

    # File uploader
    uploaded_file = st.file_uploader(
        "📁 Upload a document (PDF or Image)",
        type=['pdf', 'png', 'jpg', 'jpeg'],
        help="Supported formats: PDF, PNG, JPG"
    )

    if uploaded_file is not None:
        # Read file
        file_bytes = uploaded_file.read()

        # Convert PDF to image if needed
        if uploaded_file.type == 'application/pdf':
            with st.spinner("📄 Converting PDF to image..."):
                images = convert_from_bytes(file_bytes, dpi=200)
                image = images[0]  # Take first page
                st.info(f"ℹ️ Showing page 1 of {len(images)}")
        else:
            image = Image.open(io.BytesIO(file_bytes))

        # Display original image
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📄 Original Document")
            st.image(image, use_container_width=True)

        # Run inference
        with st.spinner("🔍 Detecting objects..."):
            detections = predict_image(model, image, conf_threshold)

        # Draw detections
        image_with_boxes = image.copy()
        image_with_boxes = draw_detections(image_with_boxes, detections)

        with col2:
            st.subheader("✅ Detections")
            st.image(image_with_boxes, use_container_width=True)

        # Statistics
        st.markdown("---")
        st.subheader("📊 Detection Statistics")

        stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)

        with stats_col1:
            st.metric("Total Detections", len(detections))
        with stats_col2:
            sig_count = sum(1 for d in detections if d['class'] == 'signature')
            st.metric("Signatures", sig_count)
        with stats_col3:
            stamp_count = sum(1 for d in detections if d['class'] == 'stamp')
            st.metric("Stamps", stamp_count)
        with stats_col4:
            qr_count = sum(1 for d in detections if d['class'] == 'qr')
            st.metric("QR Codes", qr_count)

        # Detailed results
        if detections:
            st.markdown("---")
            st.subheader("📋 Detailed Results")

            # Create table
            table_data = []
            for i, det in enumerate(detections, 1):
                table_data.append({
                    'ID': i,
                    'Type': det['class'],
                    'Confidence': f"{det['confidence']:.2%}",
                    'Position': f"({int(det['bbox']['x1'])}, {int(det['bbox']['y1'])})"
                })

            st.dataframe(table_data, use_container_width=True)

            # Download results
            import json
            results_json = json.dumps({
                'file': uploaded_file.name,
                'total_detections': len(detections),
                'detections': detections
            }, indent=2)

            st.download_button(
                label="💾 Download Results (JSON)",
                data=results_json,
                file_name=f"{uploaded_file.name}_results.json",
                mime="application/json"
            )

        else:
            st.warning("⚠️ No objects detected. Try lowering the confidence threshold.")

    else:
        # Welcome screen
        st.info("👆 Upload a document to get started!")

        # Sample results
        st.markdown("---")
        st.subheader("📈 Model Performance")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("mAP50", "0.532", help="Mean Average Precision @ IoU 0.5")
        with col2:
            st.metric("Precision", "0.866", help="Precision across all classes")
        with col3:
            st.metric("Recall", "0.509", help="Recall across all classes")

        st.markdown("---")
        st.markdown("""
        ### 🚀 How to use:
        1. Upload a PDF or image document (construction drawings, contracts, etc.)
        2. Adjust the confidence threshold if needed
        3. View detected signatures, stamps, and QR codes
        4. Download results as JSON

        ### 🎯 Use cases:
        - **Document verification:** Ensure all required signatures and stamps are present
        - **Quality control:** Automated checking of construction documents
        - **Digitization:** Extract metadata from scanned documents
        - **Compliance:** Verify regulatory requirements
        """)

if __name__ == '__main__':
    main()
