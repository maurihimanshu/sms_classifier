"""
Data cleaning utilities for SMS text preprocessing.
"""
from typing import Dict, List, Optional, Tuple
import re
from datetime import datetime
import logging
from ..config.base_config import BaseConfig

logger = logging.getLogger(__name__)

class DataCleaner:
    """
    Handles data cleaning and standardization for SMS messages.
    """

    def __init__(self, config: BaseConfig):
        self.config = config

        # Common patterns
        self._currency_symbols = r'(?:Rs\.?|INR|₹|\$|€|£)'
        self._amount_patterns = [
            rf'{self._currency_symbols}\s*(\d+(?:,\d+)*(?:\.\d{2})?)',
            r'(\d+(?:,\d+)*(?:\.\d{2})?)\s*{self._currency_symbols}'
        ]

        # Error patterns
        self._common_errors = {
            r'a/c\.?': 'account',
            r'amt\.?': 'amount',
            r'tx[nt]\.?': 'transaction',
            r'cr\.?': 'credit',
            r'dr\.?': 'debit',
            r'bal\.?': 'balance'
        }

        # Standardization mappings
        self._type_mappings = {
            'debited': 'debit',
            'withdrawn': 'debit',
            'spent': 'debit',
            'paid': 'debit',
            'credited': 'credit',
            'received': 'credit',
            'deposited': 'credit',
            'refunded': 'credit'
        }

    def clean_text(self, text: str) -> str:
        """
        Clean and standardize SMS text.

        Args:
            text: Raw SMS text

        Returns:
            Cleaned text
        """
        try:
            # Basic cleaning
            text = text.strip()
            text = self._remove_extra_whitespace(text)

            # Standardize common elements
            text = self._standardize_amounts(text)
            text = self._expand_abbreviations(text)
            text = self._standardize_transaction_type(text)
            text = self._standardize_dates(text)

            # Remove sensitive information
            text = self._mask_sensitive_data(text)

            return text

        except Exception as e:
            logger.error(f"Error cleaning text: {e}")
            return text

    def validate_sms(self, text: str) -> Tuple[bool, List[str]]:
        """
        Validate SMS text for required patterns and format.

        Args:
            text: SMS text to validate

        Returns:
            Tuple of (is_valid, list of errors)
        """
        errors = []

        # Check minimum length
        if len(text) < 10:
            errors.append("SMS text too short")

        # Check for required elements
        if not self._has_transaction_amount(text):
            errors.append("No transaction amount found")

        if not self._has_transaction_type(text):
            errors.append("No transaction type found")

        # Check for suspicious patterns
        if self._has_suspicious_patterns(text):
            errors.append("Suspicious patterns detected")

        return len(errors) == 0, errors

    def extract_structured_data(self, text: str) -> Dict:
        """
        Extract structured data from cleaned SMS text.

        Args:
            text: Cleaned SMS text

        Returns:
            Dictionary containing extracted data
        """
        data = {
            'amount': None,
            'type': None,
            'date': None,
            'account': None,
            'balance': None
        }

        # Extract amount
        amount = self._extract_amount(text)
        if amount:
            data['amount'] = amount

        # Extract transaction type
        tx_type = self._extract_transaction_type(text)
        if tx_type:
            data['type'] = tx_type

        # Extract date
        date = self._extract_date(text)
        if date:
            data['date'] = date

        # Extract account info
        account = self._extract_account(text)
        if account:
            data['account'] = account

        # Extract balance
        balance = self._extract_balance(text)
        if balance:
            data['balance'] = balance

        return data

    def _remove_extra_whitespace(self, text: str) -> str:
        """Remove extra whitespace from text."""
        return ' '.join(text.split())

    def _standardize_amounts(self, text: str) -> str:
        """Standardize amount formats."""
        # Remove commas in numbers
        text = re.sub(r'(\d),(\d)', r'\1\2', text)

        # Standardize currency format
        for pattern in self._amount_patterns:
            text = re.sub(pattern, r'RS \1', text)

        return text

    def _expand_abbreviations(self, text: str) -> str:
        """Expand common banking abbreviations."""
        text = text.lower()
        for pattern, replacement in self._common_errors.items():
            text = re.sub(pattern, replacement, text)
        return text

    def _standardize_transaction_type(self, text: str) -> str:
        """Standardize transaction type terms."""
        text = text.lower()
        for original, standard in self._type_mappings.items():
            text = re.sub(rf'\b{original}\b', standard, text)
        return text

    def _standardize_dates(self, text: str) -> str:
        """Standardize date formats."""
        # Convert DD/MM/YY to DD-MM-YYYY
        text = re.sub(
            r'(\d{1,2})[/-](\d{1,2})[/-](\d{2})(?!\d)',
            lambda m: f"{m.group(1)}-{m.group(2)}-20{m.group(3)}",
            text
        )

        # Convert DD/MM/YYYY to DD-MM-YYYY
        text = re.sub(
            r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})',
            r'\1-\2-\3',
            text
        )

        return text

    def _mask_sensitive_data(self, text: str) -> str:
        """Mask sensitive information."""
        # Mask card numbers
        text = re.sub(
            r'\b(?:\d[ -]*?){13,19}\b',
            'CARD_NUMBER',
            text
        )

        # Mask account numbers
        text = re.sub(
            r'\b(?:[A-Z]*\d+[A-Z]*){2,}\b',
            'ACCOUNT_NUMBER',
            text
        )

        return text

    def _has_transaction_amount(self, text: str) -> bool:
        """Check if text contains transaction amount."""
        return any(
            re.search(pattern, text)
            for pattern in self._amount_patterns
        )

    def _has_transaction_type(self, text: str) -> bool:
        """Check if text contains transaction type."""
        type_words = set(self._type_mappings.keys()) | set(self._type_mappings.values())
        return any(word in text.lower() for word in type_words)

    def _has_suspicious_patterns(self, text: str) -> bool:
        """Check for suspicious patterns."""
        suspicious_patterns = [
            r'(?i)password',
            r'(?i)login',
            r'(?i)click.*link',
            r'(?i)verify.*account',
            r'(?i)urgent.*action'
        ]
        return any(
            re.search(pattern, text)
            for pattern in suspicious_patterns
        )

    def _extract_amount(self, text: str) -> Optional[float]:
        """Extract transaction amount."""
        for pattern in self._amount_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    # Remove any remaining commas and convert to float
                    amount_str = match.group(1).replace(',', '')
                    return float(amount_str)
                except ValueError:
                    continue
        return None

    def _extract_transaction_type(self, text: str) -> Optional[str]:
        """Extract transaction type."""
        text = text.lower()
        for word in text.split():
            if word in self._type_mappings.values():
                return word
            if word in self._type_mappings:
                return self._type_mappings[word]
        return None

    def _extract_date(self, text: str) -> Optional[str]:
        """Extract transaction date."""
        date_patterns = [
            r'(\d{1,2}-\d{1,2}-\d{4})',
            r'(\d{1,2}/\d{1,2}/\d{4})',
            r'(\d{4}-\d{2}-\d{2})'
        ]

        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    date_str = match.group(1)
                    # Convert to standard format
                    date_obj = datetime.strptime(date_str, '%d-%m-%Y')
                    return date_obj.strftime('%Y-%m-%d')
                except ValueError:
                    continue
        return None

    def _extract_account(self, text: str) -> Optional[str]:
        """Extract account information."""
        account_patterns = [
            r'account\s+([A-Z0-9]+)',
            r'a/c\s+([A-Z0-9]+)',
            r'([A-Z0-9]+)\s+credited',
            r'([A-Z0-9]+)\s+debited'
        ]

        for pattern in account_patterns:
            match = re.search(pattern, text.lower())
            if match:
                return match.group(1).upper()
        return None

    def _extract_balance(self, text: str) -> Optional[float]:
        """Extract balance amount."""
        balance_patterns = [
            rf'balance.*?{self._currency_symbols}\s*(\d+(?:,\d+)*(?:\.\d{2})?)',
            rf'bal.*?{self._currency_symbols}\s*(\d+(?:,\d+)*(?:\.\d{2})?)'
        ]

        for pattern in balance_patterns:
            match = re.search(pattern, text.lower())
            if match:
                try:
                    balance_str = match.group(1).replace(',', '')
                    return float(balance_str)
                except ValueError:
                    continue
        return None