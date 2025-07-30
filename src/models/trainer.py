"""
Model training and fine-tuning implementation.
"""
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
import numpy as np
from tqdm import tqdm

from ..config.base_config import BaseConfig
from .classifier import TransactionClassifier
from .evaluator import ModelEvaluator

class SMSDataset(Dataset):
    """Dataset for SMS transaction data."""

    def __init__(self, texts: List[str], labels: List[Dict], tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict:
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize text
        inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Convert to tensors
        item = {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': {
                field: torch.tensor(value) 
                for field, value in label.items()
            }
        }

        if 'token_type_ids' in inputs:
            item['token_type_ids'] = inputs['token_type_ids'].squeeze()

        return item

class ModelTrainer:
    """Handles model training and fine-tuning."""

    def __init__(
        self,
        config: BaseConfig,
        model: TransactionClassifier,
        evaluator: Optional[ModelEvaluator] = None
    ):
        self.config = config
        self.model = model
        self.evaluator = evaluator or ModelEvaluator(config)

        # Initialize optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=0.01
        )

        # Training history
        self.training_history = []

    async def train(
        self,
        train_data: List[Dict],
        eval_data: Optional[List[Dict]] = None,
        num_epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        checkpoint_dir: Optional[str] = None
    ) -> Dict:
        """
        Train the model on provided data.

        Args:
            train_data: Training examples
            eval_data: Evaluation examples
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimization
            checkpoint_dir: Directory to save checkpoints

        Returns:
            Dictionary containing training metrics
        """
        # Setup training parameters
        num_epochs = num_epochs or self.config.NUM_TRAIN_EPOCHS
        batch_size = batch_size or self.config.BATCH_SIZE

        if learning_rate:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = learning_rate

        # Create datasets
        train_dataset = SMSDataset(
            [example['text'] for example in train_data],
            [example['labels'] for example in train_data],
            self.model.tokenizer
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        # Training loop
        best_metrics = None
        for epoch in range(num_epochs):
            # Train one epoch
            train_metrics = await self._train_epoch(train_loader, epoch)

            # Evaluate if data provided
            eval_metrics = None
            if eval_data:
                eval_metrics = await self.evaluator.evaluate_model(
                    self.model,
                    eval_data
                )

            # Save checkpoint if improved
            if checkpoint_dir and (
                not best_metrics or
                eval_metrics['overall']['f1'] > best_metrics['overall']['f1']
            ):
                best_metrics = eval_metrics
                await self._save_checkpoint(checkpoint_dir, epoch)

            # Record history
            self.training_history.append({
                'epoch': epoch,
                'train_metrics': train_metrics,
                'eval_metrics': eval_metrics,
                'learning_rate': self.optimizer.param_groups[0]['lr']
            })

        return {
            'train_metrics': train_metrics,
            'eval_metrics': eval_metrics,
            'history': self.training_history
        }

    async def _train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict:
        """
        Train for one epoch.

        Args:
            train_loader: DataLoader for training data
            epoch: Current epoch number

        Returns:
            Dictionary containing training metrics
        """
        self.model.train()
        total_loss = 0
        field_losses = {field: 0 for field in ['amount', 'type', 'date']}

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for batch in progress_bar:
            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                token_type_ids=batch.get('token_type_ids')
            )

            # Calculate loss for each field
            batch_loss = 0
            for field in outputs:
                field_loss = self._calculate_field_loss(
                    outputs[field],
                    batch['labels'][field]
                )
                field_losses[field] += field_loss.item()
                batch_loss += field_loss

            # Backward pass
            batch_loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=1.0
            )

            # Update weights
            self.optimizer.step()

            # Update progress
            total_loss += batch_loss.item()
            progress_bar.set_postfix({
                'loss': total_loss / (progress_bar.n + 1)
            })

        # Calculate average losses
        num_batches = len(train_loader)
        metrics = {
            'total_loss': total_loss / num_batches,
            'field_losses': {
                field: loss / num_batches
                for field, loss in field_losses.items()
            }
        }

        return metrics

    def _calculate_field_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate loss for a specific field.

        Args:
            logits: Model predictions
            labels: Ground truth labels

        Returns:
            Loss tensor
        """
        criterion = torch.nn.CrossEntropyLoss()
        return criterion(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )

    async def _save_checkpoint(
        self,
        checkpoint_dir: str,
        epoch: int
    ) -> None:
        """
        Save a model checkpoint.

        Args:
            checkpoint_dir: Directory to save checkpoint
            epoch: Current epoch number
        """
        import os

        # Create checkpoint directory if needed
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save model checkpoint
        checkpoint_path = os.path.join(
            checkpoint_dir,
            f'checkpoint_epoch_{epoch}.pt'
        )

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.model.get_config(),
            'training_history': self.training_history
        }

        torch.save(checkpoint, checkpoint_path)

    async def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load a model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path)

        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load training history
        self.training_history = checkpoint.get('training_history', [])