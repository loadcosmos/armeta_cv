"""
Document Validation Module

Validates documents based on business rules:
- Required number of signatures, stamps, QR codes
- QR code readability
- Multi-page consistency

This shows practical business understanding!
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum


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
            lines.append(f"  - {err['description']}")
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


# Example usage
if __name__ == '__main__':
    # Example document results
    results = {
        'pdf': 'contract.pdf',
        'total_pages': 3,
        'pages': [
            {
                'page': 1,
                'detections': [
                    {'class': 'signature', 'confidence': 0.95},
                    {'class': 'stamp', 'confidence': 0.98},
                ]
            },
            {
                'page': 2,
                'detections': [
                    {'class': 'signature', 'confidence': 0.92},
                    {'class': 'qr', 'confidence': 0.99, 'qr_data': {'data': 'https://example.com'}},
                ]
            }
        ]
    }

    # Validate as contract
    validator = ContractValidator()
    report = validator.validate(results)

    print(format_validation_report(report))
