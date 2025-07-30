"""
SMS text preprocessing and tokenization.
"""
from typing import Dict, List, Optional
import re
from transformers import AutoTokenizer
from ..config.base_config import BaseConfig

class SMSTokenizer:
    """
    Tokenizer for SMS transaction messages.
    """

    def __init__(self, config: BaseConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

        # Common patterns
        self.amount_pattern = r'(?:Rs\.?|INR|₹)\s*(\d+(?:,\d+)*(?:\.\d{2})?)'
        self.account_pattern = r'[Aa]/[Cc]\.?\s*([A-Z0-9]+(?:\s*[A-Z0-9]+)*)'
        self.date_pattern = r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})'

    def preprocess(self, text: str) -> str:
        """
        Preprocess SMS text.

        Args:
            text: Raw SMS text

        Returns:
            Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()

        # Normalize whitespace
        text = ' '.join(text.split())

        # Normalize account numbers
        text = re.sub(r'[Xx]+\d+', 'ACCOUNT_NUMBER', text)

        # Normalize amounts
        text = re.sub(r'(?:Rs\.?|INR|₹)\s*(\d+(?:,\d+)*(?:\.\d{2})?)', r'RS \1', text)

        # Normalize dates
        text = re.sub(r'(\d{1,2})[-/](\d{1,2})[-/](\d{2,4})', r'\1-\2-\3', text)

        return text

    def tokenize(
        self,
        text: str,
        max_length: Optional[int] = None
    ) -> Dict[str, List[int]]:
        """
        Tokenize text for model input.

        Args:
            text: Input text
            max_length: Maximum sequence length

        Returns:
            Dictionary containing tokenized inputs
        """
        # Preprocess text
        text = self.preprocess(text)

        # Tokenize
        inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=max_length or self.config.MAX_SEQ_LENGTH,
            return_tensors='pt'
        )

        return inputs

    def extract_patterns(self, text: str) -> Dict[str, Optional[str]]:
        """
        Extract common patterns from text.

        Args:
            text: Input text

        Returns:
            Dictionary containing extracted patterns
        """
        patterns = {
            'amount': None,
            'account': None,
            'date': None
        }

        # Extract amount
        amount_match = re.search(self.amount_pattern, text)
        if amount_match:
            patterns['amount'] = amount_match.group(1)

        # Extract account
        account_match = re.search(self.account_pattern, text)
        if account_match:
            patterns['account'] = account_match.group(1)

        # Extract date
        date_match = re.search(self.date_pattern, text)
        if date_match:
            patterns['date'] = date_match.group(1)

        return patterns

    def get_config(self) -> Dict:
        """
        Get tokenizer configuration.

        Returns:
            Dictionary containing configuration
        """
        return {
            'model_name': self.config.MODEL_NAME,
            'max_length': self.config.MAX_SEQ_LENGTH,
            'vocab_size': self.tokenizer.vocab_size,
            'patterns': {
                'amount': self.amount_pattern,
                'account': self.account_pattern,
                'date': self.date_pattern
            }
        }