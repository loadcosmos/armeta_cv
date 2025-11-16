"""
QR Code Decoder Module - Root Wrapper

This module provides access to the QR decoder functionality.
"""

# Import everything from the actual implementation
from kaggle.hybrid.qr_decoder import (
    QRDecoder,
    format_qr_data
)

__all__ = ['QRDecoder', 'format_qr_data']
