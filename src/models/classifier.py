"""
Transaction Classifier model implementation.
"""
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, PreTrainedModel
from ..config.base_config import BaseConfig

class TransactionClassifier(nn.Module):
    """
    BERT-based model for classifying transaction details from SMS messages.
    """

    def __init__(self, config: BaseConfig):
        super().__init__()
        self.config = config

        # Initialize BERT model
        self.bert = AutoModel.from_pretrained(config.MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

        # Token classification heads
        self.dropout = nn.Dropout(0.1)
        self.amount_classifier = nn.Linear(self.bert.config.hidden_size, 3)  # B-Amount, I-Amount, O
        self.type_classifier = nn.Linear(self.bert.config.hidden_size, 3)    # B-Type, I-Type, O
        self.date_classifier = nn.Linear(self.bert.config.hidden_size, 3)    # B-Date, I-Date, O

        # Performance tracking
        self.training_history = []
        self.performance_metrics = {}

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            input_ids: Tensor of token ids
            attention_mask: Tensor of attention mask
            token_type_ids: Optional tensor of token type ids

        Returns:
            Dictionary containing logits for each classification head
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        # Get logits for each classification head
        amount_logits = self.amount_classifier(sequence_output)
        type_logits = self.type_classifier(sequence_output)
        date_logits = self.date_classifier(sequence_output)

        return {
            'amount': amount_logits,
            'type': type_logits,
            'date': date_logits
        }

    def predict(self, text: str) -> Dict[str, Dict]:
        """
        Make predictions on a single text input.

        Args:
            text: Input SMS message

        Returns:
            Dictionary containing extracted fields and their confidence scores
        """
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.config.MAX_SEQ_LENGTH
        )

        # Get model predictions
        with torch.no_grad():
            outputs = self(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                token_type_ids=inputs.get('token_type_ids', None)
            )

        # Process predictions
        predictions = {}
        for field, logits in outputs.items():
            predictions[field] = self._process_field_predictions(
                logits[0],
                inputs['input_ids'][0],
                field
            )

        return predictions

    def _process_field_predictions(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        field: str
    ) -> Dict:
        """
        Process logits for a specific field into structured predictions.

        Args:
            logits: Model logits for the field
            input_ids: Input token ids
            field: Field name being processed

        Returns:
            Dictionary containing extracted value and confidence
        """
        # Get predicted labels
        predictions = torch.argmax(logits, dim=-1)

        # Extract tokens for the field
        tokens = []
        confidence_scores = []

        for i, (pred, token_id) in enumerate(zip(predictions, input_ids)):
            if pred == 1:  # B-Field
                tokens.append(self.tokenizer.decode([token_id]))
                confidence_scores.append(torch.softmax(logits[i], dim=-1)[1].item())
            elif pred == 2 and tokens:  # I-Field
                tokens[-1] += self.tokenizer.decode([token_id])
                confidence_scores[-1] = min(
                    confidence_scores[-1],
                    torch.softmax(logits[i], dim=-1)[2].item()
                )

        if not tokens:
            return {
                'value': None,
                'confidence': 0.0,
                'extracted_text': None
            }

        # Return the highest confidence extraction
        best_idx = confidence_scores.index(max(confidence_scores))
        return {
            'value': self._normalize_value(tokens[best_idx], field),
            'confidence': confidence_scores[best_idx],
            'extracted_text': tokens[best_idx]
        }

    def _normalize_value(self, text: str, field: str) -> str:
        """
        Normalize extracted values based on field type.

        Args:
            text: Extracted text
            field: Field type

        Returns:
            Normalized value
        """
        text = text.strip()

        if field == 'amount':
            # Remove currency symbols and convert to float
            return ''.join(c for c in text if c.isdigit() or c == '.')
        elif field == 'type':
            # Normalize transaction type
            text = text.lower()
            if any(word in text for word in ['debit', 'withdraw', 'spent']):
                return 'debit'
            elif any(word in text for word in ['credit', 'deposit', 'received']):
                return 'credit'
            return text
        elif field == 'date':
            # Keep as is for now, could add date normalization later
            return text

        return text

    @classmethod
    def from_pretrained(cls, model_state: Dict) -> 'TransactionClassifier':
        """
        Create a model instance from a saved state.

        Args:
            model_state: Dictionary containing model state and configuration

        Returns:
            Initialized model instance
        """
        model = cls(model_state['config'])
        model.load_state_dict(model_state['state_dict'])
        model.training_history = model_state.get('training_history', [])
        model.performance_metrics = model_state.get('performance_metrics', {})
        return model

    def get_config(self) -> Dict:
        """
        Get model configuration.

        Returns:
            Dictionary containing model configuration
        """
        return {
            'model_name': self.config.MODEL_NAME,
            'max_seq_length': self.config.MAX_SEQ_LENGTH,
            'hidden_size': self.bert.config.hidden_size,
            'num_labels': {
                'amount': 3,
                'type': 3,
                'date': 3
            }
        }

    def update_metrics(self, metrics: Dict) -> None:
        """
        Update model performance metrics.

        Args:
            metrics: Dictionary containing new performance metrics
        """
        self.performance_metrics.update(metrics)
        self.training_history.append({
            'metrics': metrics,
            'timestamp': torch.cuda.current_timestamp() if torch.cuda.is_available() else None
        })