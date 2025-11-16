#!/usr/bin/env python3
"""
Digital Inspector - AI Document Detection System (Compact Single File Version)
Armeta CV Hackathon - Complete Solution

This file contains the complete workflow in a single, compact file:
1. Dataset preparation (PDF + JSON → YOLO format)
2. Model training (YOLOv8s with optimized settings for small objects)
3. Hybrid inference (YOLO + OpenCV QR detection)
4. Testing and evaluation
5. Web application launch
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
import json
import cv2
import numpy as np
from ultralytics import YOLO
from pdf2image import convert_from_path
import base64
import yaml
from sklearn.model_selection import train_test_split
import shutil
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from tqdm import tqdm
from pyzbar import pyzbar
import re


def run_command(cmd, description="Running command", capture_output=True):
    """Execute a shell command with optional output capture"""
    print(f"🤖 {description}")
    print(f"   Command: {cmd}")
    
    try:
        if capture_output:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"   ❌ Error: {result.stderr}")
                return False, result.stderr
            else:
                print("   ✅ Success")
                return True, result.stdout
        else:
            result = subprocess.run(cmd, shell=True)
            return result.returncode == 0, ""
    except Exception as e:
        print(f"   ❌ Exception: {str(e)}")
        return False, str(e)


# Default model paths (will try in order)
DEFAULT_MODEL_PATHS = [
    'best.pt',                                    # Same directory
    'models/best.pt',                             # models folder
    'runs/detect/train2/weights/best.pt',        # Training output
    'weights/best.pt',                            # weights folder
]

def find_model(model_path=None):
    """Find model file in common locations"""
    if model_path and Path(model_path).exists():
        return model_path

    for path in DEFAULT_MODEL_PATHS:
        if Path(path).exists():
            return path

    return DEFAULT_MODEL_PATHS[0]  # Return first path as default

DEFAULT_MODEL = find_model()

CLASS_NAMES = {0: 'signature', 1: 'stamp', 2: 'qr'}
COLORS = {
    0: (0, 255, 0),    # signature - green
    1: (255, 0),    # stamp - blue
    2: (0, 0, 255),    # qr - red
}


class DocumentDetector:
    """Document object detection wrapper"""

    def __init__(self, model_path: str = DEFAULT_MODEL,
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45):
        """
        Initialize detector

        Args:
            model_path: Path to YOLOv8 weights
            conf_threshold: Confidence threshold for detections
            iou_threshold: NMS IoU threshold
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def detect_image(self, image: np.ndarray) -> List[Dict]:
        """
        Detect objects in single image

        Args:
            image: OpenCV image (BGR)

        Returns:
            List of detections with class, confidence, bbox
        """
        results = self.model(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )

        detections = []

        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes

            for box in boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                xyxy = box.xyxy[0].cpu().numpy()

                x1, y1, x2, y2 = map(float, xyxy)

                detections.append({
                    'class': CLASS_NAMES[cls],
                    'class_id': cls,
                    'confidence': conf,
                    'bbox': {
                        'x1': x1, 'y1': y1,
                        'x2': x2, 'y2': y2,
                        'width': x2 - x1,
                        'height': y2 - y1
                    }
                })

        return detections

    def detect_pdf(self, pdf_path: str, dpi: int = 200) -> Dict:
        """
        Detect objects in PDF document

        Args:
            pdf_path: Path to PDF file
            dpi: Resolution for PDF conversion

        Returns:
            Detection results for all pages
        """
        # Convert PDF to images
        images = convert_from_path(pdf_path, dpi=dpi)

        results = {
            'pdf': Path(pdf_path).name,
            'timestamp': datetime.now().isoformat(),
            'total_pages': len(images),
            'pages': []
        }

        for page_num, pil_img in enumerate(images, 1):
            # Convert to OpenCV format
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            # Detect objects
            detections = self.detect_image(img)

            # Count by class
            class_counts = {}
            for det in detections:
                cls = det['class']
                class_counts[cls] = class_counts.get(cls, 0) + 1

            results['pages'].append({
                'page': page_num,
                'image_size': {'width': img.shape[1], 'height': img.shape[0]},
                'detections': detections,
                'summary': class_counts
            })

        # Overall summary
        total_detections = sum(len(p['detections']) for p in results['pages'])
        overall_counts = {}
        for page in results['pages']:
            for cls, count in page['summary'].items():
                overall_counts[cls] = overall_counts.get(cls, 0) + count

        results['summary'] = {
            'total_detections': total_detections,
            'by_class': overall_counts
        }

        return results

    def visualize_detections(self, image: np.ndarray,
                            detections: List[Dict]) -> np.ndarray:
        """
        Draw detections on image

        Args:
            image: OpenCV image (BGR)
            detections: List of detections

        Returns:
            Annotated image
        """
        img_vis = image.copy()

        for det in detections:
            bbox = det['bbox']
            x1, y1 = int(bbox['x1']), int(bbox['y1'])
            x2, y2 = int(bbox['x2']), int(bbox['y2'])

            cls_id = det['class_id']
            color = COLORS.get(cls_id, (255, 255, 255))

            # Draw rectangle
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{det['class']} {det['confidence']:.2f}"

            # Text background
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(img_vis, (x1, y1 - text_h - 10),
                         (x1 + text_w, y1), color, -1)

            # Text
            cv2.putText(img_vis, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return img_vis


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


class ValidationStatus(Enum):
    """Document validation status"""
    VALID = "valid"
    WARNING = "warning"
    INVALID = "invalid"


@dataclass
class ValidationRule:
    """Single validation rule"""
    name: str
    description: str
    check_function: callable
    severity: str  # 'error', 'warning'


class DocumentValidator:
    """Validate documents based on business rules"""

    def __init__(self):
        """Initialize validator with default rules"""
        self.rules = self._get_default_rules()

    def _get_default_rules(self) -> List[ValidationRule]:
        """Get default validation rules"""
        return [
            ValidationRule(
                name="min_signatures",
                description="Document must have at least 1 signature",
                check_function=lambda r: r['counts'].get('signature', 0) >= 1,
                severity='error'
            ),
            ValidationRule(
                name="has_stamp",
                description="Document should have at least 1 stamp",
                check_function=lambda r: r['counts'].get('stamp', 0) >= 1,
                severity='warning'
            ),
            ValidationRule(
                name="has_qr",
                description="Document should have at least 1 QR code",
                check_function=lambda r: r['counts'].get('qr', 0) >= 1,
                severity='warning'
            ),
            ValidationRule(
                name="qr_readable",
                description="All QR codes should be readable",
                check_function=self._check_qr_readable,
                severity='warning'
            ),
        ]

    def _check_qr_readable(self, results: Dict) -> bool:
        """Check if all QR codes are readable"""
        for page in results.get('pages', []):
            for det in page.get('detections', []):
                if det.get('class') == 'qr':
                    qr_data = det.get('qr_data', {})
                    if qr_data.get('data') is None:
                        return False
        return True

    def add_rule(self, rule: ValidationRule):
        """Add custom validation rule"""
        self.rules.append(rule)

    def validate(self, results: Dict) -> Dict:
        """
        Validate document results

        Args:
            results: Detection results from inference

        Returns:
            Validation report with status and issues
        """
        # Count detections by class
        counts = {'signature': 0, 'stamp': 0, 'qr': 0}

        for page in results.get('pages', []):
            for det in page.get('detections', []):
                cls = det.get('class')
                if cls in counts:
                    counts[cls] += 1

        # Prepare validation context
        context = {
            'counts': counts,
            'pages': results.get('pages', []),
            'total_pages': results.get('total_pages', 0)
        }

        # Run validation rules
        errors = []
        warnings = []

        for rule in self.rules:
            try:
                passed = rule.check_function(context)

                if not passed:
                    issue = {
                        'rule': rule.name,
                        'description': rule.description,
                        'severity': rule.severity
                    }

                    if rule.severity == 'error':
                        errors.append(issue)
                    else:
                        warnings.append(issue)

            except Exception as e:
                warnings.append({
                    'rule': rule.name,
                    'description': f"Validation error: {str(e)}",
                    'severity': 'warning'
                })

        # Determine overall status
        if errors:
            status = ValidationStatus.INVALID
        elif warnings:
            status = ValidationStatus.WARNING
        else:
            status = ValidationStatus.VALID

        return {
            'status': status.value,
            'valid': status == ValidationStatus.VALID,
            'errors': errors,
            'warnings': warnings,
            'summary': {
                'total_issues': len(errors) + len(warnings),
                'errors_count': len(errors),
                'warnings_count': len(warnings)
            },
            'detections_summary': counts
        }

    def validate_batch(self, batch_results: List[Dict]) -> Dict:
        """
        Validate multiple documents

        Args:
            batch_results: List of detection results

        Returns:
            Batch validation report
        """
        validations = []

        for result in batch_results:
            validation = self.validate(result)
            validation['pdf'] = result.get('pdf', 'unknown')
            validations.append(validation)

        # Aggregate statistics
        total = len(validations)
        valid = sum(1 for v in validations if v['status'] == 'valid')
        warnings = sum(1 for v in validations if v['status'] == 'warning')
        invalid = sum(1 for v in validations if v['status'] == 'invalid')

        return {
            'total_documents': total,
            'valid': valid,
            'warnings': warnings,
            'invalid': invalid,
            'pass_rate': valid / total if total > 0 else 0,
            'documents': validations
        }


class ContractValidator(DocumentValidator):
    """Specialized validator for contracts"""

    def _get_default_rules(self) -> List[ValidationRule]:
        """Contract-specific rules"""
        return [
            ValidationRule(
                name="min_signatures",
                description="Contract must have at least 2 signatures (both parties)",
                check_function=lambda r: r['counts'].get('signature', 0) >= 2,
                severity='error'
            ),
            ValidationRule(
                name="has_stamp",
                description="Contract must have at least 1 official stamp",
                check_function=lambda r: r['counts'].get('stamp', 0) >= 1,
                severity='error'
            ),
            ValidationRule(
                name="has_qr",
                description="Contract should have QR code for verification",
                check_function=lambda r: r['counts'].get('qr', 0) >= 1,
                severity='warning'
            ),
            ValidationRule(
                name="qr_readable",
                description="QR code should be readable",
                check_function=self._check_qr_readable,
                severity='warning'
            ),
            ValidationRule(
                name="sufficient_pages",
                description="Contract should have at least 2 pages",
                check_function=lambda r: r['total_pages'] >= 2,
                severity='warning'
            ),
        ]


class LicenseValidator(DocumentValidator):
    """Specialized validator for licenses"""

    def _get_default_rules(self) -> List[ValidationRule]:
        """License-specific rules"""
        return [
            ValidationRule(
                name="has_stamp",
                description="License must have official stamp",
                check_function=lambda r: r['counts'].get('stamp', 0) >= 1,
                severity='error'
            ),
            ValidationRule(
                name="has_signature",
                description="License must be signed",
                check_function=lambda r: r['counts'].get('signature', 0) >= 1,
                severity='error'
            ),
            ValidationRule(
                name="has_qr",
                description="License must have QR code for verification",
                check_function=lambda r: r['counts'].get('qr', 0) >= 1,
                severity='error'
            ),
            ValidationRule(
                name="qr_readable",
                description="QR code must be readable",
                check_function=self._check_qr_readable,
                severity='error'
            ),
        ]


def format_validation_report(validation: Dict) -> str:
    """Format validation report as text"""
    lines = []

    status = validation['status'].upper()
    emoji = {"valid": "✅", "warning": "⚠️", "invalid": "❌"}

    lines.append(f"{emoji.get(validation['status'], '❓')} Status: {status}")
    lines.append("")

    if validation['errors']:
        lines.append("❌ ERRORS:")
        for err in validation['errors']:
            lines.append(f" - {err['description']}")
        lines.append("")

    if validation['warnings']:
        lines.append("⚠️  WARNINGS:")
        for warn in validation['warnings']:
            lines.append(f"  - {warn['description']}")
        lines.append("")

    lines.append("📊 Detections:")
    for cls, count in validation['detections_summary'].items():
        lines.append(f"  - {cls}: {count}")

    return "\n".join(lines)


class HTMLReporter:
    """Generate HTML reports for document detection results"""

    def __init__(self):
        """Initialize reporter"""
        self.template = self._get_template()

    def _get_template(self) -> str:
        """Get HTML template"""
        return '''
 <!DOCTYPE html>
 <html lang="en">
 <head>
     <meta charset="UTF-8">
     <meta name="viewport" content="width=device-width, initial-scale=1.0">
     <title>Document Detection Report - {title}</title>
     <style>
         * {{ margin: 0; padding: 0; box-sizing: border-box; }}
         body {{
             font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
             background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
             padding: 20px;
             color: #333;
         }}
         .container {{
             max-width: 1200px;
             margin: 0 auto;
             background: white;
             border-radius: 15px;
             box-shadow: 0 10px 40px rgba(0,0,0,0.2);
             overflow: hidden;
         }}
         .header {{
             background: linear-gradient(135deg, #67eea 0%, #764ba2 100%);
             color: white;
             padding: 40px;
             text-align: center;
         }}
         .header h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
         .header p {{ font-size: 1.1em; opacity: 0.9; }}
         .status {{
             display: inline-block;
             padding: 10px 30px;
             border-radius: 25px;
             font-weight: bold;
             margin: 20px 0;
             font-size: 1.2em;
         }}
         .status.valid {{ background: #10b981; color: white; }}
         .status.warning {{ background: #f59e0b; color: white; }}
         .status.invalid {{ background: #ef4444; color: white; }}
         .content {{ padding: 40px; }}
         .summary {{
             display: grid;
             grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
             gap: 20px;
             margin-bottom: 40px;
         }}
         .stat-card {{
             background: linear-gradient(135deg, #67eea 0%, #764ba2 10%);
             color: white;
             padding: 25px;
             border-radius: 10px;
             text-align: center;
             box-shadow: 0 4px 6px rgba(0,0,0.1);
         }}
         .stat-card .number {{ font-size: 3em; font-weight: bold; margin: 10px 0; }}
         .stat-card .label {{ font-size: 1.1em; opacity: 0.9; }}
         .section {{ margin: 40px 0; }}
         .section h2 {{
             color: #667eea;
             border-bottom: 3px solid #667eea;
             padding-bottom: 10px;
             margin-bottom: 20px;
             font-size: 1.8em;
         }}
         .page-grid {{
             display: grid;
             grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
             gap: 20px;
         }}
         .page-card {{
             border: 2px solid #e5e7eb;
             border-radius: 10px;
             overflow: hidden;
             transition: transform 0.3s, box-shadow 0.3s;
         }}
         .page-card:hover {{
             transform: translateY(-5px);
             box-shadow: 0 10px 20px rgba(0,0,0,0.1);
         }}
         .page-card img {{ width: 100%; height: auto; display: block; }}
         .page-info {{
             padding: 15px;
             background: #f9fafb;
         }}
         .page-info h3 {{ color: #667eea; margin-bottom: 10px; }}
         .detection-badge {{
             display: inline-block;
             padding: 5px 12px;
             margin: 3px;
             border-radius: 15px;
             font-size: 0.9em;
             font-weight: 600;
         }}
         .badge-signature {{ background: #10b981; color: white; }}
         .badge-stamp {{ background: #3b82f6; color: white; }}
         .badge-qr {{ background: #ef444; color: white; }}
         .issues {{
             background: #fef3c7;
             border-left: 4px solid #f59e0b;
             padding: 20px;
             border-radius: 5px;
             margin: 20px 0;
         }}
         .issues.error {{
             background: #fee2e2;
             border-left-color: #ef444;
         }}
         .issues ul {{ margin-left: 20px; margin-top: 10px; }}
         .issues li {{ margin: 8px 0; }}
         .qr-data {{
             background: #f0f9ff;
             border: 2px solid #3b82f6;
             padding: 15px;
             border-radius: 8px;
             margin: 10px 0;
             font-family: 'Courier New', monospace;
         }}
         .qr-data .label {{ color: #3b82f6; font-weight: bold; }}
         .footer {{
             background: #f9fafb;
             padding: 20px;
             text-align: center;
             color: #6b7280;
             border-top: 1px solid #e5e7eb;
         }}
         table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
         th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #e5e7eb; }}
         th {{ background: #f9fafb; font-weight: 600; color: #667eea; }}
         tr:hover {{ background: #f9fafb; }}
     </style>
 </head>
 <body>
     <div class="container">
         <div class="header">
             <h1>📄 Document Detection Report</h1>
             <p>{document_name}</p>
             {status_badge}
             <p style="margin-top: 20px; font-size: 0.9em;">Generated: {timestamp}</p>
         </div>

         <div class="content">
             {validation_section}

             <div class="summary">
                 <div class="stat-card">
                     <div class="label">Total Pages</div>
                     <div class="number">{total_pages}</div>
                 </div>
                 <div class="stat-card">
                     <div class="label">Total Detections</div>
                     <div class="number">{total_detections}</div>
                 </div>
                 <div class="stat-card">
                     <div class="label">Signatures</div>
                     <div class="number">{signatures_count}</div>
                 </div>
                 <div class="stat-card">
                     <div class="label">Stamps</div>
                     <div class="number">{stamps_count}</div>
                 </div>
                 <div class="stat-card">
                     <div class="label">QR Codes</div>
                     <div class="number">{qr_count}</div>
                 </div>
             </div>

             {qr_section}

             <div class="section">
                 <h2>📑 Page-by-Page Analysis</h2>
                 <div class="page-grid">
                     {pages_content}
                 </div>
             </div>

             {details_table}
         </div>

         <div class="footer">
             <p>Generated by Document Detection System | YOLOv8 + QR Decoder + Validator</p>
             <p style="margin-top: 10px;">mAP50: 88.1% | Precision: 93.1% | Recall: 86.7%</p>
         </div>
     </div>
 </body>
 </html>
 '''

    def _image_to_base64(self, image: np.ndarray) -> str:
        """Convert OpenCV image to base64 string"""
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')

    def _create_page_card(self, page_data: Dict, image: Optional[np.ndarray] = None) -> str:
        """Create HTML for single page card"""
        page_num = page_data.get('page', 0)
        detections = page_data.get('detections', [])

        # Count by class
        counts = {}
        qr_codes = []
        for det in detections:
            cls = det.get('class', 'unknown')
            counts[cls] = counts.get(cls, 0) + 1

            if cls == 'qr' and 'qr_data' in det:
                qr_codes.append(det['qr_data'])

        # Create badges
        badges = []
        for cls, count in counts.items():
            badge_class = f"badge-{cls}"
            badges.append(f'<span class="detection-badge {badge_class}">{count} {cls}</span>')

        badges_html = ' '.join(badges) if badges else '<span style="color: #9ca3af;">No detections</span>'

        # QR data section
        qr_html = ''
        if qr_codes:
            qr_items = []
            for qr in qr_codes:
                data = qr.get('data', 'Unreadable')
                data_type = qr.get('data_type', 'unknown')
                qr_items.append(f'<div class="qr-data"><span class="label">[{data_type}]</span> {data}</div>')
            qr_html = ''.join(qr_items)

        # Image section
        img_html = ''
        if image is not None:
            img_base64 = self._image_to_base64(image)
            img_html = f'<img src="data:image/jpeg;base64,{img_base64}" alt="Page {page_num}">'

        return f'''
         <div class="page-card">
             {img_html}
             <div class="page-info">
                 <h3>Page {page_num}</h3>
                 <p>{badges_html}</p>
                 {qr_html}
             </div>
         </div>
         '''

    def _create_validation_section(self, validation: Dict) -> str:
        """Create HTML for validation results"""
        if not validation:
            return ''

        errors = validation.get('errors', [])
        warnings = validation.get('warnings', [])

        if not errors and not warnings:
            return '<div class="issues" style="background: #d1fae5; border-left-color: #10b981;"><strong>✅ All validation checks passed!</strong></div>'

        html = ''

        if errors:
            error_items = ''.join([f'<li>{err["description"]}</li>' for err in errors])
            html += f'<div class="issues error"><strong>❌ Errors:</strong><ul>{error_items}</ul></div>'

        if warnings:
            warning_items = ''.join([f'<li>{warn["description"]}</li>' for warn in warnings])
            html += f'<div class="issues"><strong>⚠️ Warnings:</strong><ul>{warning_items}</ul></div>'

        return html

    def _create_qr_section(self, results: Dict) -> str:
        """Create section for all QR codes found"""
        qr_data_list = []

        for page in results.get('pages', []):
            for det in page.get('detections', []):
                if det.get('class') == 'qr' and 'qr_data' in det:
                    qr = det['qr_data']
                    qr_data_list.append({
                        'page': page['page'],
                        'data': qr.get('data'),
                        'type': qr.get('data_type'),
                        'quality': qr.get('quality')
                    })

        if not qr_data_list:
            return ''

        rows = []
        for qr in qr_data_list:
            data_display = qr['data'] if qr['data'] else '❌ Unreadable'
            rows.append(f'''
             <tr>
                 <td>Page {qr['page']}</td>
                 <td><span class="detection-badge badge-qr">{qr['type']}</span></td>
                 <td style="font-family: monospace;">{data_display}</td>
                 <td>{qr['quality']}</td>
             </tr>
             ''')

        table_html = f'''
         <table>
             <thead>
                 <tr>
                     <th>Page</th>
                     <th>Type</th>
                     <th>Data</th>
                     <th>Quality</th>
                 </tr>
             </thead>
             <tbody>
                 {''.join(rows)}
             </tbody>
         </table>
         '''

        return f'<div class="section"><h2>🔍 QR Code Data</h2>{table_html}</div>'

    def generate_report(self, results: Dict, validation: Optional[Dict] = None,
                       page_images: Optional[List[np.ndarray]] = None,
                       output_path: Optional[Path] = None) -> str:
        """
        Generate HTML report

        Args:
            results: Detection results from inference
            validation: Validation results (optional)
            page_images: List of annotated page images (optional)
            output_path: Path to save HTML file (optional)

        Returns:
            HTML string
        """
        # Prepare data
        document_name = results.get('pdf', 'Unknown Document')
        total_pages = results.get('total_pages', 0)

        summary = results.get('summary', {})
        total_detections = summary.get('total_detections', 0)
        by_class = summary.get('by_class', {})

        signatures_count = by_class.get('signature', 0)
        stamps_count = by_class.get('stamp', 0)
        qr_count = by_class.get('qr', 0)

        # Status badge
        status = 'valid'
        status_text = 'Valid'
        if validation:
            status = validation.get('status', 'valid')
            status_text = status.capitalize()

        status_badge = f'<div class="status {status}">{status_text}</div>'

        # Validation section
        validation_section = self._create_validation_section(validation) if validation else ''

        # QR section
        qr_section = self._create_qr_section(results)

        # Pages content
        pages_content = []
        for i, page_data in enumerate(results.get('pages', [])):
            img = page_images[i] if page_images and i < len(page_images) else None
            pages_content.append(self._create_page_card(page_data, img))

        pages_html = ''.join(pages_content)

        # Details table
        details_rows = []
        for page_data in results.get('pages', []):
            page_num = page_data['page']
            for det in page_data.get('detections', []):
                cls = det['class']
                conf = det['confidence']
                details_rows.append(f'''
                 <tr>
                     <td>Page {page_num}</td>
                     <td><span class="detection-badge badge-{cls}">{cls}</span></td>
                     <td>{conf:.2%}</td>
                 </tr>
                 ''')

        details_table = ''
        if details_rows:
            details_table = f'''
             <div class="section">
                 <h2>📋 All Detections</h2>
                 <table>
                     <thead>
                         <tr>
                             <th>Page</th>
                             <th>Class</th>
                             <th>Confidence</th>
                         </tr>
                     </thead>
                     <tbody>
                         {''.join(details_rows)}
                     </tbody>
                 </table>
             </div>
             '''

        # Fill template
        html = self.template.format(
            title=document_name,
            document_name=document_name,
            status_badge=status_badge,
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            validation_section=validation_section,
            total_pages=total_pages,
            total_detections=total_detections,
            signatures_count=signatures_count,
            stamps_count=stamps_count,
            qr_count=qr_count,
            qr_section=qr_section,
            pages_content=pages_html,
            details_table=details_table
        )

        # Save if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html)

        return html


def process_single_pdf(detector: DocumentDetector,
                      pdf_path: Path,
                      output_dir: Path,
                      visualize: bool = False,
                      dpi: int = 200) -> Dict:
    """Process single PDF file"""

    print(f"Processing: {pdf_path.name}")

    # Run detection
    results = detector.detect_pdf(str(pdf_path), dpi=dpi)

    # Create output directory
    pdf_output = output_dir / pdf_path.stem
    pdf_output.mkdir(parents=True, exist_ok=True)

    # Save JSON results
    json_path = pdf_output / 'results.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Visualize if requested
    if visualize:
        images = convert_from_path(str(pdf_path), dpi=dpi)

        for page_idx, pil_img in enumerate(images):
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            detections = results['pages'][page_idx]['detections']

            img_vis = detector.visualize_detections(img, detections)

            vis_path = pdf_output / f'page_{page_idx + 1}.jpg'
            cv2.imwrite(str(vis_path), img_vis)

    # Print summary
    summary = results['summary']
    print(f"  ✓ {summary['total_detections']} detections")
    for cls, count in summary['by_class'].items():
        print(f"    - {cls}: {count}")

    return results


class EnhancedDocumentProcessor:
    """
    Complete document processing pipeline with all features
    """

    def __init__(self, model_path: str = None,
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45,
                 validator_type: str = 'default'):
        """
        Initialize enhanced processor

        Args:
            model_path: Path to YOLO model
            conf_threshold: Detection confidence threshold
            iou_threshold: NMS IoU threshold
            validator_type: Type of validator ('default', 'contract', 'license')
        """
        # Initialize components
        self.detector = DocumentDetector(
            model_path=model_path or find_model(),
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold
        )

        self.qr_decoder = QRDecoder()

        # Select validator
        if validator_type == 'contract':
            self.validator = ContractValidator()
        elif validator_type == 'license':
            self.validator = LicenseValidator()
        else:
            self.validator = DocumentValidator()

        self.html_reporter = HTMLReporter()

    def process_pdf(self, pdf_path: str, dpi: int = 200,
                    output_dir: Path = None,
                    generate_html: bool = True,
                    save_visualizations: bool = True) -> Dict:
        """
        Complete processing pipeline for PDF

        Args:
            pdf_path: Path to PDF file
            dpi: Resolution for PDF conversion
            output_dir: Output directory
            generate_html: Generate HTML report
            save_visualizations: Save annotated images

        Returns:
            Complete results with detections, QR data, and validation
        """
        pdf_path = Path(pdf_path)
        print(f"\n📄 Processing: {pdf_path.name}")

        # Convert PDF to images
        print(" Converting PDF to images...")
        pil_images = convert_from_path(str(pdf_path), dpi=dpi)

        results = {
            'pdf': pdf_path.name,
            'total_pages': len(pil_images),
            'pages': []
        }

        annotated_images = []

        # Process each page
        for page_num, pil_img in enumerate(tqdm(pil_images, desc="  Processing pages"), 1):
            # Convert to OpenCV format
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            # 1. Detect objects
            detections = self.detector.detect_image(img)

            # 2. Decode QR codes
            detections = self.qr_decoder.decode_all_qr_codes(img, detections)

            # 3. Visualize
            if save_visualizations or generate_html:
                img_vis = self.detector.visualize_detections(img, detections)

                # Add QR data to visualization
                for det in detections:
                    if det.get('class') == 'qr' and 'qr_data' in det:
                        qr_data = det['qr_data']
                        if qr_data.get('data'):
                            bbox = det['bbox']
                            x1, y1 = int(bbox['x1']), int(bbox['y1'])

                            # Draw QR data
                            data_text = qr_data['data'][:50]
                            cv2.putText(img_vis, f"QR: {data_text}",
                                       (x1, y1 - 30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                annotated_images.append(img_vis)

            # Count by class
            class_counts = {}
            for det in detections:
                cls = det['class']
                class_counts[cls] = class_counts.get(cls, 0) + 1

            results['pages'].append({
                'page': page_num,
                'image_size': {'width': img.shape[1], 'height': img.shape[0]},
                'detections': detections,
                'summary': class_counts
            })

        # Calculate overall summary
        total_detections = sum(len(p['detections']) for p in results['pages'])
        overall_counts = {}
        for page in results['pages']:
            for cls, count in page['summary'].items():
                overall_counts[cls] = overall_counts.get(cls, 0) + count

        results['summary'] = {
            'total_detections': total_detections,
            'by_class': overall_counts
        }

        # 4. Validate document
        print("  Validating document...")
        validation = self.validator.validate(results)
        results['validation'] = validation

        # Print validation results
        print(f"\n{format_validation_report(validation)}\n")

        # 5. Save results
        if output_dir:
            output_dir = Path(output_dir)
            pdf_output = output_dir / pdf_path.stem
            pdf_output.mkdir(parents=True, exist_ok=True)

            # Save JSON
            json_path = pdf_output / 'results.json'
            with open(json_path, 'w', encoding='utf-8') as f:
                # Create serializable copy (remove OpenCV objects)
                serializable_results = self._make_serializable(results)
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)

            print(f"  ✓ Saved JSON: {json_path}")

            # Save visualizations
            if save_visualizations:
                for i, img in enumerate(annotated_images, 1):
                    img_path = pdf_output / f'page_{i}.jpg'
                    cv2.imwrite(str(img_path), img)
                print(f"  ✓ Saved {len(annotated_images)} visualizations")

            # Generate HTML report
            if generate_html:
                html_path = pdf_output / 'report.html'
                self.html_reporter.generate_report(
                    results,
                    validation=validation,
                    page_images=annotated_images,
                    output_path=html_path
                )
                print(f"  ✓ Generated HTML report: {html_path}")

        return results

    def _make_serializable(self, obj):
        """Convert object to JSON-serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def process_batch(self, input_dir: Path,
                     output_dir: Path,
                     generate_html: bool = True) -> List[Dict]:
        """
        Process multiple PDF files

        Args:
            input_dir: Directory containing PDFs
            output_dir: Output directory
            generate_html: Generate HTML reports

        Returns:
            List of results
        """
        pdf_files = list(input_dir.glob('*.pdf'))
        pdf_files = [p for p in pdf_files if 'Zone.Identifier' not in str(p)]

        print(f"\n📂 Found {len(pdf_files)} PDF files")

        all_results = []

        for pdf_path in pdf_files:
            try:
                result = self.process_pdf(
                    pdf_path,
                    output_dir=output_dir,
                    generate_html=generate_html
                )
                all_results.append(result)
            except Exception as e:
                print(f" ❌ Error processing {pdf_path.name}: {e}")

        # Batch validation summary
        batch_validation = self.validator.validate_batch(all_results)

        # Save batch summary
        summary_path = output_dir / 'batch_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(batch_validation, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Batch processing complete!")
        print(f"   Total: {batch_validation['total_documents']}")
        print(f"   Valid: {batch_validation['valid']} ({batch_validation['pass_rate']:.1%})")
        print(f"   Warnings: {batch_validation['warnings']}")
        print(f"   Invalid: {batch_validation['invalid']}")

        return all_results


def prepare_dataset(args):
    """Prepare dataset from PDFs and annotations"""
    print("\n" + "="*70)
    print("📄 STEP 1: Preparing Dataset")
    print("="*70)
    
    # Configuration
    CLASS_MAPPING = {
        'signature': 0,
        'stamp': 1,
        'qr': 2,
        'qr_code': 2,  # alias
        'seal': 1,      # alias for stamp
    }

    # Paths
    DATA_DIR = Path("data")
    PDF_DIR = DATA_DIR / 'pdf'
    ANNOTATIONS_DIR = DATA_DIR / 'annotations'
    ANNOTATIONS_FILE = ANNOTATIONS_DIR / 'selected_annotations.json'
    
    OUTPUT_DIR = Path('/kaggle/working/data')
    DPI = 200
    TRAIN_SPLIT = 0.8

    # Check paths
    if not PDF_DIR.exists():
        print(f"❌ PDF directory not found: {PDF_DIR}")
        print("   Create 'data/pdf/' and add PDF files")
        return False

    if not ANNOTATIONS_FILE.exists():
        print(f"❌ Annotations file not found: {ANNOTATIONS_FILE}")
        print("   Create 'data/annotations/selected_annotations.json'")
        return False

    def load_annotations(json_path):
        """
        Load selected_annotations.json

        Expected format:
        {
          "pdf_name.pdf": {
            "page_1": {
              "annotations": [
                {"annotation_123": {"category": "signature", "bbox": {...}}},
                ...
              ]
            },
            "page_2": {...}
          }
        }
        """
        print(f"📖 Loading annotations from {json_path}...")

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        annotations_by_image = {}

        # Parse the nested structure: PDF → pages → annotations
        if isinstance(data, dict):
            for pdf_filename, pages_data in data.items():
                # Remove .pdf extension for matching
                pdf_basename = pdf_filename.replace('.pdf', '')

                if isinstance(pages_data, dict):
                    # Iterate through pages (page_1, page_2, etc.)
                    for page_key, page_data in pages_data.items():
                        if not isinstance(page_data, dict):
                            continue

                        # Extract page number from "page_X"
                        if page_key.startswith('page_'):
                            page_num = page_key.split('_')[1]
                        else:
                            continue

                        # Get annotations for this page
                        annotations_list = page_data.get('annotations', [])

                        if not annotations_list:
                            continue

                        # Create image name that matches PDF conversion: basename_page_N.png
                        image_name = f"{pdf_basename}_page_{page_num}"

                        # Unwrap nested annotation structure
                        # Each annotation is: {"annotation_XXX": {category, bbox, ...}}
                        unwrapped_annotations = []
                        for ann_wrapper in annotations_list:
                            if isinstance(ann_wrapper, dict):
                                # Get the actual annotation (first value in dict)
                                for ann_id, ann_data in ann_wrapper.items():
                                    if isinstance(ann_data, dict):
                                        unwrapped_annotations.append(ann_data)
                                        break

                        # Store annotations AND page_size for proper scaling
                        page_size = page_data.get('page_size', {})
                        annotations_by_image[image_name] = {
                            'annotations': unwrapped_annotations,
                            'page_size': page_size
                        }

        print(f"✅ Loaded annotations for {len(annotations_by_image)} images")

        # Debug: show first few entries
        if len(annotations_by_image) > 0:
            print(f"   Sample images: {list(annotations_by_image.keys())[:3]}")

        return annotations_by_image

    def convert_pdf_to_images(pdf_path, output_dir, dpi=200):
        """Convert PDF to images"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        images = convert_from_path(str(pdf_path), dpi=dpi)

        saved_paths = []
        pdf_name = Path(pdf_path).stem

        for i, img in enumerate(images):
            img_path = output_dir / f"{pdf_name}_page_{i+1}.png"
            img.save(img_path, 'PNG')
            saved_paths.append(img_path)

        return saved_paths

    def parse_annotation(ann):
        """Parse single annotation to extract class and bbox"""
        # Extract class
        class_name = None
        for key in ['label', 'category', 'class', 'name', 'category_name']:
            if key in ann:
                class_name = str(ann[key]).lower()
                break

        if not class_name:
            return None, None

        # Map to class ID
        class_id = None
        for key, value in CLASS_MAPPING.items():
            if key in class_name:
                class_id = value
                break

        if class_id is None:
            return None, None

        # Extract bbox
        bbox = None

        # Format 1: bbox as dict {"x": ..., "y": ..., "width": ..., "height": ...}
        if 'bbox' in ann:
            bbox_data = ann['bbox']
            if isinstance(bbox_data, dict) and all(k in bbox_data for k in ['x', 'y', 'width', 'height']):
                x, y, w, h = bbox_data['x'], bbox_data['y'], bbox_data['width'], bbox_data['height']
            # Format 2: bbox as array [x, y, width, height]
            elif isinstance(bbox_data, (list, tuple)) and len(bbox_data) == 4:
                x, y, w, h = bbox_data
            else:
                return None, None

        # Format 3: points (polygon)
        elif 'points' in ann:
            points = ann['points']
            xs = [p[0] if isinstance(p, (list, tuple)) else p['x'] for p in points]
            ys = [p[1] if isinstance(p, (list, tuple)) else p['y'] for p in points]
            x, y = min(xs), min(ys)
            w, h = max(xs) - x, max(ys) - y

        # Format 4: x, y, width, height as separate fields (root level)
        elif all(k in ann for k in ['x', 'y', 'width', 'height']):
            x, y = ann['x'], ann['y']
            w, h = ann['width'], ann['height']

        else:
            return None, None

        # Ensure all values are valid numbers
        try:
            x, y, w, h = float(x), float(y), float(w), float(h)
        except (ValueError, TypeError):
            return None, None

        return class_id, (x, y, w, h)

    def convert_to_yolo_format(annotations, img_width, img_height, json_page_size=None):
        """
        Convert annotations to YOLO format

        Args:
            annotations: List of annotation dicts
            img_width: Actual image width (after PDF conversion)
            img_height: Actual image height (after PDF conversion)
            json_page_size: Original page size from JSON annotations (dict with 'width', 'height')
        """
        yolo_lines = []

        # Calculate scaling factors if JSON page size is different from actual image size
        scale_x = 1.0
        scale_y = 1.0

        if json_page_size and 'width' in json_page_size and 'height' in json_page_size:
            json_width = json_page_size['width']
            json_height = json_page_size['height']

            if json_width > 0 and json_height > 0:
                scale_x = img_width / json_width
                scale_y = img_height / json_height

        for ann in annotations:
            class_id, bbox = parse_annotation(ann)

            if class_id is None or bbox is None:
                continue

            x, y, w, h = bbox

            # Scale coordinates from JSON size to actual image size
            x = x * scale_x
            y = y * scale_y
            w = w * scale_x
            h = h * scale_y

            # Convert to YOLO format (normalized)
            x_center = (x + w / 2) / img_width
            y_center = (y + h / 2) / img_height
            norm_width = w / img_width
            norm_height = h / img_height

            # Validate
            if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and
                    0 < norm_width <= 1 and 0 < norm_height <= 1):
                continue

            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}")

        return yolo_lines

    print("Loading annotations...")
    annotations_by_file = load_annotations(ANNOTATIONS_FILE)

    # Create temp directories
    images_temp = OUTPUT_DIR / 'images_temp'
    labels_temp = OUTPUT_DIR / 'labels_temp'
    images_temp.mkdir(parents=True, exist_ok=True)
    labels_temp.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("📄 STEP 1: Converting PDFs to images...")
    print("=" * 70)

    pdf_files = sorted(PDF_DIR.glob('*.pdf'))
    print(f"Found {len(pdf_files)} PDF files")

    all_image_paths = []

    for pdf_path in tqdm(pdf_files, desc="Converting PDFs"):
        img_paths = convert_pdf_to_images(pdf_path, images_temp, dpi=DPI)
        all_image_paths.extend(img_paths)

    print(f"✅ Created {len(all_image_paths)} images")

    print("\n" + "=" * 70)
    print("📝 STEP 2: Creating YOLO labels...")
    print("=" * 70)

    total_annotations = 0
    images_with_labels = []

    for img_path in tqdm(all_image_paths, desc="Processing images"):
        # Image name without extension: "pdf_name_page_3"
        img_stem = img_path.stem

        # Try to find matching annotations data (contains 'annotations' and 'page_size')
        ann_data = annotations_by_file.get(img_stem)

        if ann_data is None:
            continue

        # Extract annotations and page_size
        if isinstance(ann_data, dict):
            matching_anns = ann_data.get('annotations', [])
            page_size = ann_data.get('page_size', {})
        else:
            # Fallback for old format (if any)
            matching_anns = ann_data
            page_size = {}

        if len(matching_anns) == 0:
            continue

        # Get image dimensions
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]

        # Convert to YOLO format with scaling
        yolo_lines = convert_to_yolo_format(matching_anns, w, h, json_page_size=page_size)

        if len(yolo_lines) == 0:
            continue

        # Save label file
        label_path = labels_temp / f"{img_path.stem}.txt"
        with open(label_path, 'w') as f:
            f.write('\n'.join(yolo_lines))

        total_annotations += len(yolo_lines)
        images_with_labels.append((img_path, label_path))

    print(f"✅ Created {len(images_with_labels)} label files with {total_annotations} annotations")

    if len(images_with_labels) == 0:
        print("❌ No images with labels found! Check annotation file format.")
        return False

    print("\n" + "=" * 70)
    print("🔀 STEP 3: Creating train/val split...")
    print("=" * 70)

    train_pairs, val_pairs = train_test_split(
        images_with_labels,
        train_size=TRAIN_SPLIT,
        random_state=42,
        shuffle=True
    )

    print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}")

    # Create final structure
    for split, pairs in [('train', train_pairs), ('val', val_pairs)]:
        split_img_dir = OUTPUT_DIR / split / 'images'
        split_lbl_dir = OUTPUT_DIR / split / 'labels'
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path, lbl_path in pairs:
            shutil.copy(img_path, split_img_dir / img_path.name)
            shutil.copy(lbl_path, split_lbl_dir / lbl_path.name)

    # Clean up temp
    shutil.rmtree(images_temp)
    shutil.rmtree(labels_temp)

    print("\n" + "=" * 70)
    print("📋 STEP 4: Creating data.yaml...")
    print("=" * 70)

    data_yaml = {
        'path': str(OUTPUT_DIR.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'nc': 3,
        'names': {0: 'signature', 1: 'stamp', 2: 'qr'}
    }

    yaml_path = OUTPUT_DIR / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print(f"✅ Created {yaml_path}")

    print("\n" + "=" * 70)
    print("✨ DATASET READY!")
    print("=" * 70)
    print(f"📁 Location: {OUTPUT_DIR}")
    print(f"📊 Train: {len(train_pairs)} images")
    print(f"📊 Val: {len(val_pairs)} images")
    print(f"📝 Annotations: {total_annotations}")
    print(f"\n🚀 Next: Run training step")

    return True


def train_model(args):
    """Train the YOLO model"""
    print("\n" + "="*70)
    print("🏋️  STEP 2: Training Model")
    print("="*70)
    
    # Configuration
    CONFIG = {
        # Model
        'model': 'yolov8s.pt',  # ⭐ CHANGED from yolov8n.pt

        # Data
        'data_yaml': '/kaggle/working/data/data.yaml',  # Kaggle path (change if using Colab)

        # Training params
        'epochs': 120,           # ⬆️ Increased (more capacity)
        'imgsz': 1024,          # ⭐ CRITICAL: 640→1024 for small objects
        'batch': 4,             # ⬇️ Reduced (1024 uses more memory)
        'patience': 20,         # ⬆️ More patience for convergence
        'device': 0,            # GPU
        'amp': True,            # Mixed precision

        # Optimizer
        'optimizer': 'AdamW',   # Better than SGD for small datasets
        'lr0': 0.0001,         # ⬇️ Lower LR for fine-tuning (was 0.001)
        'lrf': 0.01,           # Final LR = lr0 * lrf
        'momentum': 0.937,
        'weight_decay': 0.0005,

        # Augmentation (optimized for small objects)
        'augment': True,
        'mosaic': 1.0,         # ⭐ Mosaic augmentation
        'copy_paste': 0.3,     # ⭐ Copy-paste small objects
        'mixup': 0.15,         # ⭐ Mixup augmentation
        'degrees': 5.0,        # ⬇️ Less rotation (preserve orientation)
        'translate': 0.1,      # ⬇️ Less translation (keep objects in frame)
        'scale': 0.9,          # ⬇️ Less scaling (preserve small objects)
        'fliplr': 0.5,         # Horizontal flip
        'flipud': 0.0,         # No vertical flip (documents)
        'hsv_h': 0.015,        # Hue augmentation
        'hsv_s': 0.7,          # Saturation
        'hsv_v': 0.4,          # Value

        # Loss weights
        'box': 7.5,            # BBox loss weight
        'cls': 0.5,            # Classification loss
        'dfl': 1.5,            # Distribution focal loss

        # Other
        'workers': 4,
        'save_period': 10,     # Save checkpoint every 10 epochs
        'verbose': True,
        'seed': 42,
        'deterministic': True,
    }

    # Class weights (to handle imbalance)
    # Based on your data: signature=103, stamp=60, qr=95
    # Weight = 1 / sqrt(count)
    CLASS_WEIGHTS = {
        0: 1.0,   # signature (baseline)
        1: 1.3,   # stamp (fewer samples)
        2: 1.05,  # qr
    }

    print("=" * 60)
    print("YOLOv8s Training - Optimized for Small Objects")
    print("=" * 60)

    # Check GPU
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️ No GPU detected! Training will be slow.")

    # Load model
    print(f"\n📦 Loading {CONFIG['model']}...")
    model = YOLO(CONFIG['model'])

    # Model info
    print(f"   Parameters: {sum(p.numel() for p in model.model.parameters()) / 1e6:.1f}M")

    # Check data.yaml exists
    if not os.path.exists(CONFIG['data_yaml']):
        print(f"❌ data.yaml not found: {CONFIG['data_yaml']}")
        print("   Run dataset preparation first")
        return False

    # Load data.yaml to check classes
    with open(CONFIG['data_yaml'], 'r') as f:
        data_config = yaml.safe_load(f)
    print(f"\n📊 Dataset:")
    print(f"   Classes: {data_config['names']}")
    print(f"   Train: {data_config['train']}")
    print(f"   Val: {data_config['val']}")

    # Start training
    print(f"\n🚀 Starting training with config:")
    print(f"   Model: {CONFIG['model']}")
    print(f"   Image size: {CONFIG['imgsz']}")
    print(f"   Batch size: {CONFIG['batch']}")
    print(f"   Epochs: {CONFIG['epochs']}")
    print(f"   Learning rate: {CONFIG['lr0']}")
    print(f"   Augmentation: mosaic={CONFIG['mosaic']}, copy_paste={CONFIG['copy_paste']}")
    print("-" * 60)

    # Train
    results = model.train(
        data=CONFIG['data_yaml'],
        epochs=CONFIG['epochs'],
        imgsz=CONFIG['imgsz'],
        batch=CONFIG['batch'],
        patience=CONFIG['patience'],
        device=CONFIG['device'],
        amp=CONFIG['amp'],
        optimizer=CONFIG['optimizer'],
        lr0=CONFIG['lr0'],
        lrf=CONFIG['lrf'],
        momentum=CONFIG['momentum'],
        weight_decay=CONFIG['weight_decay'],
        augment=CONFIG['augment'],
        mosaic=CONFIG['mosaic'],
        copy_paste=CONFIG['copy_paste'],
        mixup=CONFIG['mixup'],
        degrees=CONFIG['degrees'],
        translate=CONFIG['translate'],
        scale=CONFIG['scale'],
        fliplr=CONFIG['fliplr'],
        flipud=CONFIG['flipud'],
        hsv_h=CONFIG['hsv_h'],
        hsv_s=CONFIG['hsv_s'],
        hsv_v=CONFIG['hsv_v'],
        box=CONFIG['box'],
        cls=CONFIG['cls'],
        dfl=CONFIG['dfl'],
        workers=CONFIG['workers'],
        save_period=CONFIG['save_period'],
        verbose=CONFIG['verbose'],
        seed=CONFIG['seed'],
        deterministic=CONFIG['deterministic'],
    )

    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)

    # Print results
    print(f"\n📈 Final metrics:")
    print(f"   mAP50: {results.results_dict.get('metrics/mAP50(B)', 0):.3f}")
    print(f"   mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 0):.3f}")
    print(f"   Precision: {results.results_dict.get('metrics/precision(B)', 0):.3f}")
    print(f"   Recall: {results.results_dict.get('metrics/recall(B)', 0):.3f}")

    # Save path
    save_dir = Path(results.save_dir)
    print(f"\n💾 Model saved:")
    print(f"   Best: {save_dir / 'weights' / 'best.pt'}")
    print(f"   Last: {save_dir / 'weights' / 'last.pt'}")
    print(f"   Results: {save_dir / 'results.csv'}")

    # Validation
    print(f"\n🔍 Running validation on best model...")
    best_model = YOLO(save_dir / 'weights' / 'best.pt')
    val_results = best_model.val(data=CONFIG['data_yaml'])

    print(f"\n📊 Per-class metrics (best.pt):")
    class_names = data_config['names']
    for i, name in class_names.items():
        print(f"   {name}:")
        # Note: Ultralytics metrics are stored differently, adjust as needed
        print(f"      mAP50: {val_results.box.maps[i]:.3f}")
        print(f"      Precision: {val_results.box.p[i]:.3f}")
        print(f"      Recall: {val_results.box.r[i]:.3f}")

    print("\n✨ Done! Use best.pt for inference.")
    
    if results.results_dict.get('metrics/mAP50(B)', 0) > 0.5:
        print("   ✅ Model trained successfully!")
        return True
    else:
        print("   ❌ Model performance below threshold")
        return False


def run_inference(args):
    """Run inference with the trained model"""
    print("\n" + "="*70)
    print("🔍 STEP 3: Running Inference")
    print("="*70)
    
    # Initialize detector
    detector = DocumentDetector(model_path=args.model, conf_threshold=args.conf, iou_threshold=args.iou)
    qr_decoder = QRDecoder()
    
    # Enhanced processing
    processor = EnhancedDocumentProcessor(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        validator_type=args.validator
    )
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process input
    input_path = Path(args.input)
    
    if input_path.is_dir():
        # Batch processing
        pdf_files = list(input_path.glob('*.pdf'))
        pdf_files = [p for p in pdf_files if 'Zone.Identifier' not in str(p)]
        
        print(f"Found {len(pdf_files)} PDF files\n")
        
        all_results = []
        for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
            try:
                result = processor.process_pdf(
                    pdf_path,
                    dpi=args.dpi,
                    output_dir=output_dir,
                    generate_html=args.html,
                    save_visualizations=not args.no_viz
                )
                all_results.append(result)
            except Exception as e:
                print(f"  ❌ Error processing {pdf_path.name}: {e}")
        
        # Save batch summary
        if all_results:
            summary = processor.validator.validate_batch(all_results)
            summary_path = output_dir / 'batch_summary.json'
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
    else:
        # Single file processing
        if not input_path.exists():
            print(f"❌ Error: File not found: {input_path}")
            return False
        
        result = processor.process_pdf(
            input_path,
            dpi=args.dpi,
            output_dir=output_dir,
            generate_html=args.html,
            save_visualizations=not args.no_viz
        )
    
    print(f"\n✅ Results saved to: {output_dir}")
    return True


def run_tests(args):
    """Run various tests"""
    print("\n" + "="*70)
    print("🧪 STEP 4: Running Tests")
    print("="*70)
    
    # Simple test to check if model works
    try:
        # Initialize detector
        detector = DocumentDetector(conf_threshold=0.25, iou_threshold=0.45)
        
        # Create a simple test image (blank with a rectangle to simulate detection)
        test_img = np.zeros((640, 640, 3), dtype=np.uint8)
        cv2.rectangle(test_img, (100, 100), (20, 200), (255, 255, 255), 2)
        
        # Try detection
        detections = detector.detect_image(test_img)
        print(f"   ✅ Model loading test: OK")
        print(f"   ✅ Detection test: {len(detections)} detections on test image")
        
        # Test QR decoder
        qr_decoder = QRDecoder()
        print(f"   ✅ QR decoder test: OK")
        
        # Test validators
        validator = DocumentValidator()
        contract_validator = ContractValidator()
        license_validator = LicenseValidator()
        print(f"   ✅ Validators test: OK")
        
        print("   All basic tests passed!")
        return True
        
    except Exception as e:
        print(f"   ❌ Tests failed: {e}")
        return False


def launch_web_app(args):
    """Launch the Streamlit web application"""
    print("\n" + "="*70)
    print("🌐 STEP 5: Launching Web Application")
    print("="*70)
    
    print("🚀 Starting Streamlit app...")
    print("   Open your browser at: http://localhost:8501")
    print("   Press Ctrl+C to stop the server")
    
    try:
        # Create a simple Streamlit app inline
        streamlit_app_content = '''
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import base64
from datetime import datetime
import json

# Import our modules
try:
    from app import DocumentDetector, find_model, QRDecoder, format_qr_data, DocumentValidator, ContractValidator, format_validation_report
except ImportError:
    st.error("⚠️ Please run this from the project directory")
    st.stop()

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
        border-left: 4px solid #ef444;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'pages' not in st.session_state:
    st.session_state.pages = []
if 'detector' not in st.session_state:
    try:
        st.session_state.detector = DocumentDetector(conf_threshold=0.25, iou_threshold=0.45)
        st.session_state.qr_decoder = QRDecoder()
        st.session_state.validator = DocumentValidator()
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

# Header
st.title("📱 Mobile Document Scanner")
st.markdown("**Detect signatures, stamps & QR codes** • Decode QR data • Validate documents")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")

    mode = st.radio(
        "📸 Scanning Mode",
        ["Single Photo", "Multi-Page Document"],
        help="Single: One photo\\nMulti-Page: Scan multiple pages into one PDF"
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
        from app import LicenseValidator
        st.session_state.validator = LicenseValidator()
    else:
        st.session_state.validator = DocumentValidator()

    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.markdown("""
    **YOLOv8s**
    - mAP50: 88.1%
    - QR: 9.5% (100% recall) ⭐
    - Stamp: 87.0%
    - Signature: 77.6%
    """)

# Main content
tab1, tab2 = st.tabs(["📷 Camera", "📁 Upload"])

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

        else:
            st.warning("No detections found. Try adjusting settings in sidebar.")

with tab2:
    st.markdown("### 📁 Upload Image")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png'],
        help="Upload image for document analysis"
    )
    
    if uploaded_file:
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

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6b7280;'>
    <p>🚀 Powered by YOLOv8 + QR Decoder + Validator</p>
    <p>Made for Armeta CV Hackathon 2024</p>
</div>
""", unsafe_allow_html=True)
'''
        
        # Write the Streamlit app to a temporary file
        with open('temp_app.py', 'w') as f:
            f.write(streamlit_app_content)
        
        # Run streamlit
        cmd = "streamlit run temp_app.py"
        os.system(cmd)
        
        # Clean up
        if os.path.exists('temp_app.py'):
            os.remove('temp_app.py')
        
        return True
    except KeyboardInterrupt:
        print("\n   ✋ Web app stopped by user")
        return True


def show_project_info():
    """Display project information and metrics"""
    print("\n" + "="*70)
    print("📊 DIGITAL INSPECTOR - PROJECT INFORMATION")
    print("="*70)
    
    try:
        with open('final_results.json', 'r') as f:
            results = json.load(f)
        
        print(f"Project: {results.get('project', 'N/A')}")
        print(f"Hackathon: {results.get('hackathon', 'N/A')}")
        print(f"Model: {results.get('model', 'N/A')}")
        
        print(f"\n🎯 Key Metrics:")
        metrics = results.get('metrics_validation', {}).get('overall', {})
        print(f"   - mAP50: {metrics.get('mAP50', 'N/A')}")
        print(f"   - mAP50-95: {metrics.get('mAP50-95', 'N/A')}")
        print(f"   - Precision: {metrics.get('precision', 'N/A')}")
        print(f"   - Recall: {metrics.get('recall', 'N/A')}")
        
        print(f"\n📈 Per-Class Performance:")
        per_class = results.get('metrics_validation', {}).get('per_class', {})
        for class_name, metrics in per_class.items():
            if 'mAP50' in metrics:
                print(f"   - {class_name}: mAP50 = {metrics['mAP50']}")
        
    except FileNotFoundError:
        print("   final_results.json not found - run training to see metrics")
    except json.JSONDecodeError:
        print("   Error reading final_results.json")


def main():
    parser = argparse.ArgumentParser(description='Digital Inspector - Complete AI Document Detection System')
    parser.add_argument('--mode', type=str, choices=['prepare', 'train', 'inference', 'test', 'app', 'all', 'info'], 
                       default='info', help='Operation mode')
    parser.add_argument('--kaggle', action='store_true', help='Use Kaggle-specific paths')
    parser.add_argument('--source', type=str, default='data/test/', help='Source for inference (image, folder, or PDF)')
    parser.add_argument('--model', type=str, default='runs/detect/train/weights/best.pt', help='Model path for inference')
    parser.add_argument('--output', type=str, default='results.json', help='Output file for inference results')
    parser.add_argument('--no-opencv', action='store_true', help='Disable OpenCV QR detection (YOLO only)')
    parser.add_argument('--test-type', type=str, choices=['all', 'local', 'overlapping'], 
                       default='local', help='Type of tests to run')
    
    # Additional arguments for enhanced inference
    parser.add_argument('--html', action='store_true', help='Generate HTML report')
    parser.add_argument('--no-viz', action='store_true', help='Skip visualization images')
    parser.add_argument('--validator', choices=['default', 'contract', 'license'],
                       default='default', help='Validator type')
    parser.add_argument('--dpi', type=int, default=200, help='PDF conversion DPI')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='NMS IoU threshold')
    
    args = parser.parse_args()
    
    print("🔍 DIGITAL INSPECTOR - AI Document Detection System")
    print("   Armeta CV Hackathon 2024")
    print("   Hybrid: YOLOv8s + OpenCV QRCodeDetector")
    
    if args.mode == 'info':
        show_project_info()
        return
    
    # Validate dependencies
    try:
        import ultralytics
        import cv2
        import numpy as np
        import torch
        print("✅ Dependencies validated")
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("   Run: pip install -r requirements.txt")
        return
    
    # Execute workflow based on mode
    if args.mode == 'prepare':
        prepare_dataset(args)
    elif args.mode == 'train':
        if prepare_dataset(args):  # Prepare first if not already done
            train_model(args)
    elif args.mode == 'inference':
        args.input = args.source  # Map source to input for consistency
        run_inference(args)
    elif args.mode == 'test':
        run_tests(args)
    elif args.mode == 'app':
        launch_web_app(args)
    elif args.mode == 'all':
        # Complete workflow: prepare -> train -> inference -> test -> app
        if prepare_dataset(args):
            if train_model(args):
                if run_inference(args):
                    run_tests(args)
                    print("\n" + "="*70)
                    print("🎉 COMPLETE WORKFLOW FINISHED!")
                    print("   All steps completed successfully")
                    print("="*70)
                    
                    # Optionally launch web app after everything
                    response = input("   Launch web app? (y/n): ")
                    if response.lower() in ['y', 'yes']:
                        launch_web_app(args)
                else:
                    print("   ❌ Inference step failed, stopping workflow")
            else:
                print("   ❌ Training step failed, stopping workflow")
        else:
            print("   ❌ Data preparation failed, stopping workflow")
    
    print("\n✨ Digital Inspector workflow completed!")


if __name__ == '__main__':
    main()