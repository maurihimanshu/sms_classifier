"""
Version control implementation for model management.
"""
from typing import Dict, Optional, List
from datetime import datetime
import uuid
import logging
from sqlalchemy.orm import Session

from ..config.version_config import VersionConfig
from ..config.base_config import BaseConfig
from .storage import ModelStorage
from .validation import VersionValidation
from .lock import LockManager

logger = logging.getLogger(__name__)

class ModelVersionControl:
    def __init__(self, config: BaseConfig):
        self.storage = ModelStorage()
        self.validation = VersionValidation()
        self.lock_manager = LockManager()
        self.config = config

    async def save_version(self, model: 'TransactionClassifier', metadata: Dict):
        """
        Save a new model version with comprehensive state and metadata.

        Args:
            model: The model instance to version
            metadata: Additional metadata about the version

        Returns:
            str: The version ID of the saved model
        """
        version_id = self._generate_version_id()

        try:
            # Acquire distributed lock for atomic operation
            async with self.lock_manager.acquire(f"model_version_{version_id}"):
                # Prepare model state
                model_state = await self._prepare_model_state(model)

                # Validate model state
                if not await self.validation.validate_model_state(model_state):
                    raise ValidationError("Invalid model state")

                # Create version info
                version_info = self._create_version_info(version_id, model_state, metadata)

                # Save version atomically
                await self._save_version_atomic(version_id, model_state, version_info)

                # Create recovery point
                await self._create_recovery_point(version_id)

            return version_id

        except Exception as e:
            logger.error(f"Failed to save version {version_id}: {e}")
            await self._handle_version_save_failure(version_id)
            raise

    async def _prepare_model_state(self, model: 'TransactionClassifier') -> Dict:
        """Prepare complete model state for versioning."""
        return {
            'state_dict': model.state_dict(),
            'config': model.config,
            'training_history': model.training_history,
            'performance_metrics': model.performance_metrics,
            'preprocessing_config': model.preprocessor.config,
            'tokenizer_config': model.tokenizer.get_config()
        }

    def _create_version_info(self, version_id: str, model_state: Dict, metadata: Dict) -> Dict:
        """Create comprehensive version metadata."""
        return {
            'version_id': version_id,
            'timestamp': datetime.utcnow(),
            'metadata': metadata,
            'performance': model_state['performance_metrics'],
            'training_data_hash': self._calculate_data_hash(),
            'dependencies': self._get_dependency_versions(),
            'git_commit': self._get_git_commit(),
            'environment': self._get_environment_info(),
            'status': 'draft'
        }

    async def _save_version_atomic(self, version_id: str, model_state: Dict, version_info: Dict):
        """Save version data atomically."""
        async with self.storage.transaction() as session:
            # Save model state
            await self.storage.save_model_state(session, version_id, model_state)

            # Store version metadata
            await self.storage.store_version_info(session, version_info)

            # Update version index
            await self._update_version_index(session, version_id, version_info)

    async def get_version(self, version_id: str) -> Optional[Dict]:
        """
        Retrieve a specific version with validation.

        Args:
            version_id: The ID of the version to retrieve

        Returns:
            Optional[Dict]: The version data if found and valid
        """
        version_data = await self.storage.get_model_state(version_id)
        if version_data:
            if await self.validation.validate_version_integrity(version_data):
                return version_data
            else:
                logger.error(f"Version integrity check failed for {version_id}")
                return None
        return None

    async def list_versions(self, 
                          limit: int = 10, 
                          include_metrics: bool = True) -> List[Dict]:
        """
        List available versions with optional metrics.

        Args:
            limit: Maximum number of versions to return
            include_metrics: Whether to include performance metrics

        Returns:
            List[Dict]: List of version information
        """
        versions = await self.storage.get_versions(limit)
        if include_metrics:
            for version in versions:
                version['metrics'] = await self._get_version_metrics(version['version_id'])
        return versions

    async def transition_version(self, version_id: str, new_status: str):
        """
        Transition a version to a new status.

        Args:
            version_id: The version to transition
            new_status: The new status
        """
        async with self.lock_manager.acquire(f"version_transition_{version_id}"):
            current_status = await self.storage.get_version_status(version_id)

            if not VersionConfig.is_valid_transition(current_status, new_status):
                raise ValueError(f"Invalid transition from {current_status} to {new_status}")

            if new_status == 'stable':
                if not await self._validate_version_stability(version_id):
                    raise ValidationError("Version not stable enough for transition")

            await self.storage.update_version_status(version_id, new_status)

    async def _validate_version_stability(self, version_id: str) -> bool:
        """Validate version stability based on metrics."""
        stability_metrics = await self._get_stability_metrics(version_id)
        return all([
            stability_metrics['accuracy_stability'] > VersionConfig.MIN_ACCURACY_THRESHOLD,
            stability_metrics['latency_stability'] > 0.90,
            stability_metrics['error_rate'] < VersionConfig.MAX_ERROR_RATE,
            stability_metrics['memory_usage_stable']
        ])

    def _generate_version_id(self) -> str:
        """Generate a unique version ID."""
        return f"v_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    async def _create_recovery_point(self, version_id: str):
        """Create a recovery point for safe rollback."""
        recovery_data = {
            'version_id': version_id,
            'timestamp': datetime.utcnow(),
            'previous_version': await self.get_current_active_version(),
            'recovery_scripts': self._generate_recovery_scripts()
        }
        await self.storage.store_recovery_point(recovery_data)

    async def _handle_version_save_failure(self, version_id: str):
        """Handle failure during version save."""
        try:
            await self.storage.cleanup_failed_version(version_id)
        except Exception as e:
            logger.error(f"Failed to cleanup version {version_id}: {e}")