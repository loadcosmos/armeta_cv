"""
QR Code Decoder Module

Decodes QR codes detected by YOLO and extracts their content.
This adds significant value by not just detecting, but also reading QR data.
"""

import cv2
import numpy as np
from typing import Dict, Optional, List
from pyzbar import pyzbar
import re


class QRDecoder:
    """Decode QR codes from image regions"""

    def __init__(self):
        """Initialize QR decoder"""
        self.cv_detector = cv2.QRCodeDetector()

    def decode_region(self, image: np.ndarray, bbox: Dict) -> Optional[Dict]:
        """
        Decode QR code from bounding box region

        Args:
            image: Full image (BGR)
            bbox: Bounding box dict with x1, y1, x2, y2

        Returns:
            Dict with decoded data or None
        """
        # Extract region
        x1 = int(bbox['x1'])
        y1 = int(bbox['y1'])
        x2 = int(bbox['x2'])
        y2 = int(bbox['y2'])

        # Add padding for better detection
        padding = 20
        h, w = image.shape[:2]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)

        region = image[y1:y2, x1:x2]

        if region.size == 0:
            return None

        # Try pyzbar first (more reliable)
        decoded_data = self._decode_with_pyzbar(region)
        if decoded_data:
            return decoded_data

        # Fallback to OpenCV
        decoded_data = self._decode_with_opencv(region)
        return decoded_data

    def _decode_with_pyzbar(self, region: np.ndarray) -> Optional[Dict]:
        """Decode using pyzbar library"""
        try:
            decoded_objects = pyzbar.decode(region)

            if decoded_objects:
                obj = decoded_objects[0]  # Take first QR code
                data = obj.data.decode('utf-8', errors='ignore')

                return {
                    'method': 'pyzbar',
                    'type': obj.type,
                    'data': data,
                    'data_type': self._classify_data(data),
                    'quality': 'high'
                }
        except Exception as e:
            pass

        return None

    def _decode_with_opencv(self, region: np.ndarray) -> Optional[Dict]:
        """Decode using OpenCV QRCodeDetector"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

            # Try direct detection
            retval, decoded_info, points, _ = self.cv_detector.detectAndDecodeMulti(gray)

            if retval and decoded_info and decoded_info[0]:
                data = decoded_info[0]

                return {
                    'method': 'opencv',
                    'type': 'QR_CODE',
                    'data': data,
                    'data_type': self._classify_data(data),
                    'quality': 'medium'
                }
        except Exception as e:
            pass

        return None

    def _classify_data(self, data: str) -> str:
        """Classify QR code data type"""
        if not data:
            return 'empty'

        # URL patterns
        url_pattern = r'^https?://'
        if re.match(url_pattern, data):
            return 'url'

        # Email pattern
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if re.match(email_pattern, data):
            return 'email'

        # Phone pattern
        phone_pattern = r'^[\+]?[(]?[0-9]{1,4}[)]?[-\s\.]?[(]?[0-9]{1,4}[)]?[-\s\.]?[0-9]{1,9}$'
        if re.match(phone_pattern, data.replace(' ', '')):
            return 'phone'

        # JSON-like pattern
        if data.strip().startswith('{') and data.strip().endswith('}'):
            return 'json'

        # Number pattern
        if data.replace('.', '').replace('-', '').isdigit():
            return 'number'

        # Default
        return 'text'

    def decode_all_qr_codes(self, image: np.ndarray, detections: List[Dict]) -> List[Dict]:
        """
        Decode all QR codes in detections list

        Args:
            image: Full image
            detections: List of detection dicts

        Returns:
            Updated detections with QR data
        """
        updated_detections = []

        for det in detections:
            det_copy = det.copy()

            # Only process QR codes
            if det.get('class') == 'qr':
                decoded = self.decode_region(image, det['bbox'])

                if decoded:
                    det_copy['qr_data'] = decoded
                else:
                    det_copy['qr_data'] = {
                        'method': 'none',
                        'type': 'unknown',
                        'data': None,
                        'data_type': 'unreadable',
                        'quality': 'low'
                    }

            updated_detections.append(det_copy)

        return updated_detections


def format_qr_data(qr_data: Dict) -> str:
    """Format QR data for display"""
    if not qr_data or qr_data.get('data') is None:
        return "❌ Unreadable"

    data = qr_data['data']
    data_type = qr_data.get('data_type', 'text')

    if data_type == 'url':
        return f"🔗 {data[:50]}..." if len(data) > 50 else f"🔗 {data}"
    elif data_type == 'email':
        return f"📧 {data}"
    elif data_type == 'phone':
        return f"📱 {data}"
    elif data_type == 'number':
        return f"🔢 {data}"
    else:
        return f"📄 {data[:50]}..." if len(data) > 50 else f"📄 {data}"


# Example usage
if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python qr_decoder.py <image_path>")
        sys.exit(1)

    # Load image
    image = cv2.imread(sys.argv[1])
    if image is None:
        print("Error: Could not load image")
        sys.exit(1)

    # Create decoder
    decoder = QRDecoder()

    # Detect and decode all QR codes using OpenCV
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    qr_detector = cv2.QRCodeDetector()
    retval, decoded_info, points, _ = qr_detector.detectAndDecodeMulti(gray)

    if retval and points is not None:
        print(f"Found {len(points)} QR codes:")
        for i, (info, pts) in enumerate(zip(decoded_info, points), 1):
            print(f"\nQR #{i}:")
            print(f"  Data: {info}")
            print(f"  Type: {decoder._classify_data(info)}")
    else:
        print("No QR codes found")
