"""
Model evaluation and performance metrics calculation.
"""
from typing import Dict, List, Tuple, Optional
import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from ..config.base_config import BaseConfig
from .classifier import TransactionClassifier

class ModelEvaluator:
    """
    Evaluates model performance and validates predictions.
    """

    def __init__(self, config: BaseConfig):
        self.config = config
        self.metrics_history = []

    async def evaluate_model(
        self,
        model: TransactionClassifier,
        eval_data: List[Dict],
        threshold: float = 0.5
    ) -> Dict:
        """
        Evaluate model performance on a dataset.

        Args:
            model: The model to evaluate
            eval_data: List of evaluation examples
            threshold: Confidence threshold for predictions

        Returns:
            Dictionary containing performance metrics
        """
        model.eval()
        all_metrics = {
            'amount': {'true': [], 'pred': [], 'confidence': []},
            'type': {'true': [], 'pred': [], 'confidence': []},
            'date': {'true': [], 'pred': [], 'confidence': []}
        }

        with torch.no_grad():
            for example in eval_data:
                # Get model predictions
                predictions = model.predict(example['text'])

                # Collect predictions and ground truth
                for field in ['amount', 'type', 'date']:
                    if predictions[field]['confidence'] >= threshold:
                        all_metrics[field]['pred'].append(predictions[field]['value'])
                        all_metrics[field]['true'].append(example[field])
                        all_metrics[field]['confidence'].append(predictions[field]['confidence'])

        # Calculate metrics for each field
        metrics = {}
        for field in all_metrics:
            if all_metrics[field]['true']:
                field_metrics = self._calculate_field_metrics(
                    all_metrics[field]['true'],
                    all_metrics[field]['pred'],
                    all_metrics[field]['confidence']
                )
                metrics[field] = field_metrics

        # Add to history
        self.metrics_history.append({
            'metrics': metrics,
            'timestamp': torch.cuda.current_timestamp() if torch.cuda.is_available() else None
        })

        return metrics

    def _calculate_field_metrics(
        self,
        true_values: List[str],
        pred_values: List[str],
        confidences: List[float]
    ) -> Dict:
        """
        Calculate metrics for a specific field.

        Args:
            true_values: Ground truth values
            pred_values: Predicted values
            confidences: Prediction confidences

        Returns:
            Dictionary containing field-specific metrics
        """
        # Calculate precision, recall, f1
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_values,
            pred_values,
            average='weighted'
        )

        # Calculate confusion matrix
        cm = confusion_matrix(true_values, pred_values)

        # Calculate accuracy
        accuracy = (cm.diagonal().sum() / cm.sum()) if cm.sum() > 0 else 0

        # Calculate confidence metrics
        mean_confidence = np.mean(confidences)
        confidence_std = np.std(confidences)

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'confusion_matrix': cm.tolist(),
            'mean_confidence': mean_confidence,
            'confidence_std': confidence_std
        }

    async def validate_predictions(
        self,
        predictions: Dict[str, Dict],
        threshold: float = 0.5
    ) -> Tuple[bool, Dict]:
        """
        Validate model predictions.

        Args:
            predictions: Model predictions
            threshold: Confidence threshold

        Returns:
            Tuple of (is_valid, validation_info)
        """
        validation_info = {
            'is_valid': True,
            'fields': {},
            'overall_confidence': 0.0
        }

        field_confidences = []

        # Validate each field
        for field, pred in predictions.items():
            field_valid, field_info = self._validate_field(field, pred, threshold)
            validation_info['fields'][field] = field_info
            validation_info['is_valid'] &= field_valid

            if pred['confidence'] is not None:
                field_confidences.append(pred['confidence'])

        # Calculate overall confidence
        validation_info['overall_confidence'] = (
            np.mean(field_confidences) if field_confidences else 0.0
        )

        return validation_info['is_valid'], validation_info

    def _validate_field(
        self,
        field: str,
        prediction: Dict,
        threshold: float
    ) -> Tuple[bool, Dict]:
        """
        Validate a specific field prediction.

        Args:
            field: Field name
            prediction: Field prediction
            threshold: Confidence threshold

        Returns:
            Tuple of (is_valid, validation_info)
        """
        validation_info = {
            'confidence': prediction['confidence'],
            'value': prediction['value'],
            'errors': []
        }

        # Check confidence threshold
        if prediction['confidence'] < threshold:
            validation_info['errors'].append(f"Confidence below threshold: {prediction['confidence']:.2f} < {threshold}")

        # Validate value format
        if prediction['value'] is not None:
            if field == 'amount':
                try:
                    float(prediction['value'])
                except ValueError:
                    validation_info['errors'].append("Invalid amount format")

            elif field == 'type':
                if prediction['value'] not in ['debit', 'credit']:
                    validation_info['errors'].append("Invalid transaction type")

            elif field == 'date':
                # Basic date format check
                if not self._is_valid_date_format(prediction['value']):
                    validation_info['errors'].append("Invalid date format")

        is_valid = len(validation_info['errors']) == 0
        return is_valid, validation_info

    def _is_valid_date_format(self, date_str: str) -> bool:
        """Check if string matches expected date format."""
        import re
        # Add more date format patterns as needed
        date_patterns = [
            r'\d{1,2}-\d{1,2}-\d{2,4}',
            r'\d{1,2}/\d{1,2}/\d{2,4}',
            r'\d{4}-\d{2}-\d{2}',
            r'\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}'
        ]
        return any(re.match(pattern, date_str) for pattern in date_patterns)

    def get_metrics_summary(self, last_n: Optional[int] = None) -> Dict:
        """
        Get summary of historical metrics.

        Args:
            last_n: Number of recent evaluations to include

        Returns:
            Dictionary containing metrics summary
        """
        if not self.metrics_history:
            return {}

        history = self.metrics_history[-last_n:] if last_n else self.metrics_history

        summary = {}
        for field in ['amount', 'type', 'date']:
            field_metrics = []
            for entry in history:
                if field in entry['metrics']:
                    field_metrics.append(entry['metrics'][field])

            if field_metrics:
                summary[field] = {
                    'accuracy_mean': np.mean([m['accuracy'] for m in field_metrics]),
                    'accuracy_std': np.std([m['accuracy'] for m in field_metrics]),
                    'f1_mean': np.mean([m['f1'] for m in field_metrics]),
                    'f1_std': np.std([m['f1'] for m in field_metrics]),
                    'confidence_mean': np.mean([m['mean_confidence'] for m in field_metrics]),
                    'confidence_std': np.mean([m['confidence_std'] for m in field_metrics])
                }

        return summary