"""
Document Validation Module - Root Wrapper

This module provides access to document validation functionality.
"""

# Import everything from the actual implementation
from kaggle.hybrid.validator import (
    ValidationStatus,
    ValidationRule,
    DocumentValidator,
    ContractValidator,
    LicenseValidator,
    format_validation_report
)

__all__ = [
    'ValidationStatus',
    'ValidationRule',
    'DocumentValidator',
    'ContractValidator',
    'LicenseValidator',
    'format_validation_report'
]
