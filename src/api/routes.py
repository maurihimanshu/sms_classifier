"""
API routes implementation.
"""
from typing import Dict, Optional
from datetime import datetime
import uuid
import logging
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse

from ..config.base_config import BaseConfig
from ..models.classifier import TransactionClassifier
from ..models.evaluator import ModelEvaluator
from ..preprocessing.data_cleaner import DataCleaner
from .schemas import (
    PredictionRequest,
    PredictionResponse,
    FeedbackRequest,
    FeedbackResponse,
    ErrorResponse,
    HealthResponse,
    PerformanceResponse
)

# Initialize router
router = APIRouter()

# Initialize logger
logger = logging.getLogger(__name__)

class APIContext:
    """Holds API context and dependencies."""

    def __init__(self):
        self.config = BaseConfig()
        self.model = None
        self.evaluator = None
        self.data_cleaner = None
        self.start_time = datetime.utcnow()

    async def initialize(self):
        """Initialize API dependencies."""
        if not self.model:
            try:
                self.model = TransactionClassifier(self.config)
                self.evaluator = ModelEvaluator(self.config)
                self.data_cleaner = DataCleaner(self.config)
            except Exception as e:
                logger.error(f"Failed to initialize API context: {e}")
                raise HTTPException(
                    status_code=500,
                    detail="Failed to initialize service"
                )

# Create API context
api_context = APIContext()

async def get_context() -> APIContext:
    """Dependency to get API context."""
    await api_context.initialize()
    return api_context

@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def predict_sms(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    context: APIContext = Depends(get_context)
) -> PredictionResponse:
    """
    Predict transaction details from SMS message.
    """
    try:
        # Clean and validate text
        cleaned_text = context.data_cleaner.clean_text(request.message)
        is_valid, errors = context.data_cleaner.validate_sms(cleaned_text)

        if not is_valid:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Invalid SMS message",
                    "errors": errors
                }
            )

        # Get model predictions
        predictions = context.model.predict(cleaned_text)

        # Validate predictions
        is_valid, validation_info = await context.evaluator.validate_predictions(
            predictions
        )

        # Generate response
        response = PredictionResponse(
            prediction_id=f"pred_{uuid.uuid4().hex[:8]}",
            message_id=f"msg_{uuid.uuid4().hex[:8]}",
            model_version=context.model.get_config()['model_name'],
            timestamp=datetime.utcnow(),
            extracted_fields=predictions,
            overall_confidence=validation_info['overall_confidence']
        )

        # Schedule background tasks
        background_tasks.add_task(
            _log_prediction,
            response.prediction_id,
            request.message,
            predictions
        )

        return response

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Failed to process prediction",
                "error": str(e)
            }
        )

@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def submit_feedback(
    request: FeedbackRequest,
    background_tasks: BackgroundTasks,
    context: APIContext = Depends(get_context)
) -> FeedbackResponse:
    """
    Submit feedback for a prediction.
    """
    try:
        # Generate feedback ID
        feedback_id = f"fb_{uuid.uuid4().hex[:8]}"

        # Calculate quality score
        quality_score = _calculate_feedback_quality(request.corrections)

        # Schedule feedback processing
        background_tasks.add_task(
            _process_feedback,
            feedback_id,
            request.prediction_id,
            request.corrections
        )

        return FeedbackResponse(
            feedback_id=feedback_id,
            status="accepted",
            quality_score=quality_score
        )

    except Exception as e:
        logger.error(f"Feedback error: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Failed to process feedback",
                "error": str(e)
            }
        )

@router.get(
    "/health",
    response_model=HealthResponse,
    responses={
        500: {"model": ErrorResponse}
    }
)
async def health_check(
    context: APIContext = Depends(get_context)
) -> HealthResponse:
    """
    Check service health status.
    """
    try:
        uptime = (datetime.utcnow() - context.start_time).total_seconds()

        return HealthResponse(
            status="healthy",
            version="1.0.0",
            model_version=context.model.get_config()['model_name'],
            uptime=uptime,
            components={
                "model": "healthy",
                "database": "healthy",
                "cache": "healthy"
            }
        )

    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Health check failed",
                "error": str(e)
            }
        )

@router.get(
    "/metrics",
    response_model=PerformanceResponse,
    responses={
        500: {"model": ErrorResponse}
    }
)
async def get_metrics(
    context: APIContext = Depends(get_context)
) -> PerformanceResponse:
    """
    Get model performance metrics.
    """
    try:
        # Get metrics summary
        metrics = await context.evaluator.get_metrics_summary()

        return PerformanceResponse(
            model_version=context.model.get_config()['model_name'],
            timestamp=datetime.utcnow(),
            overall_metrics=metrics['overall'],
            field_metrics=metrics['fields'],
            recent_predictions=metrics.get('recent_predictions', 0),
            feedback_rate=metrics.get('feedback_rate', 0.0)
        )

    except Exception as e:
        logger.error(f"Metrics error: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Failed to get metrics",
                "error": str(e)
            }
        )

async def _log_prediction(
    prediction_id: str,
    message: str,
    predictions: Dict
) -> None:
    """Log prediction details for monitoring."""
    try:
        logger.info(
            "Prediction logged",
            extra={
                "prediction_id": prediction_id,
                "message_length": len(message),
                "confidence": predictions.get('overall_confidence', 0.0)
            }
        )
    except Exception as e:
        logger.error(f"Failed to log prediction: {e}")

async def _process_feedback(
    feedback_id: str,
    prediction_id: str,
    corrections: Dict
) -> None:
    """Process feedback in background."""
    try:
        logger.info(
            "Processing feedback",
            extra={
                "feedback_id": feedback_id,
                "prediction_id": prediction_id,
                "corrections_count": len(corrections)
            }
        )
        # TODO: Implement feedback processing logic

    except Exception as e:
        logger.error(f"Failed to process feedback: {e}")

def _calculate_feedback_quality(corrections: Dict) -> float:
    """Calculate feedback quality score."""
    if not corrections:
        return 0.0

    # Calculate based on correction completeness and consistency
    total_fields = len(corrections)
    valid_corrections = sum(
        1 for c in corrections.values()
        if not c.is_correct and c.corrected_value is not None
    )

    return min(valid_corrections / total_fields, 1.0)