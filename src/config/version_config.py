"""
Version management configuration for the SMS Transaction Classifier system.
"""
from typing import Dict
from datetime import timedelta

class VersionConfig:
    # Version Management
    MAX_VERSIONS_RETAINED = 5
    MIN_VERSIONS_RETAINED = 2
    VERSION_RETENTION_DAYS = 30

    # Storage Configuration
    STORAGE_REDUNDANCY = 2
    BACKUP_INTERVAL_HOURS = 24
    COMPRESSION_ENABLED = True

    # Validation Thresholds
    MIN_ACCURACY_THRESHOLD = 0.95
    MAX_LATENCY_MS = 100
    MAX_MEMORY_USAGE_MB = 2048

    # Stability Requirements
    STABILITY_MONITORING_HOURS = 48
    MIN_INFERENCE_SAMPLES = 1000
    MAX_ERROR_RATE = 0.02

    # Recovery Configuration
    MAX_RECOVERY_TIME_SECONDS = 300
    RECOVERY_VERIFICATION_REQUIRED = True
    AUTO_ROLLBACK_ENABLED = True

    # Version Status Transitions
    VERSION_STATES = {
        'draft': ['testing'],
        'testing': ['stable', 'archived'],
        'stable': ['archived'],
        'archived': []
    }

    # Monitoring Windows
    STABILITY_WINDOW = timedelta(hours=STABILITY_MONITORING_HOURS)
    PERFORMANCE_WINDOW = timedelta(hours=1)
    ALERT_WINDOW = timedelta(minutes=5)

    @classmethod
    def is_valid_transition(cls, current_state: str, new_state: str) -> bool:
        """Check if state transition is valid."""
        if current_state not in cls.VERSION_STATES:
            return False
        return new_state in cls.VERSION_STATES[current_state]

    @classmethod
    def get_monitoring_windows(cls) -> Dict[str, timedelta]:
        """Get monitoring window configurations."""
        return {
            'stability': cls.STABILITY_WINDOW,
            'performance': cls.PERFORMANCE_WINDOW,
            'alert': cls.ALERT_WINDOW
        }