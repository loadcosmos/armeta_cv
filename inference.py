"""
YOLO Document Detector - Root Wrapper

This module provides access to YOLO document detection functionality.
"""

# Import everything from the actual implementation
from kaggle.hybrid.inference import (
    DocumentDetector,
    YOLODocumentDetector,
    find_model,
    DEFAULT_MODEL,
    CLASS_NAMES,
    COLORS
)

__all__ = [
    'DocumentDetector',
    'YOLODocumentDetector',
    'find_model',
    'DEFAULT_MODEL',
    'CLASS_NAMES',
    'COLORS'
]
