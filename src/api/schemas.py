"""
API request and response schemas.
"""
from typing import Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field, validator
from enum import Enum

class TransactionType(str, Enum):
    """Transaction type enumeration."""
    DEBIT = "debit"
    CREDIT = "credit"

class PredictionRequest(BaseModel):
    """
    Request schema for SMS prediction.
    """
    message: str = Field(..., description="SMS message text")
    metadata: Optional[Dict] = Field(
        default={},
        description="Optional metadata about the message"
    )

    @validator('message')
    def validate_message(cls, v):
        if not v or len(v.strip()) < 10:
            raise ValueError("Message too short")
        return v.strip()

class FieldPrediction(BaseModel):
    """
    Prediction for a specific field.
    """
    value: Optional[str] = Field(None, description="Extracted value")
    confidence: float = Field(..., description="Confidence score")
    extracted_text: Optional[str] = Field(
        None,
        description="Original text that was extracted"
    )
    position: Optional[Dict[str, int]] = Field(
        None,
        description="Position in text where value was found"
    )

class PredictionResponse(BaseModel):
    """
    Response schema for SMS prediction.
    """
    prediction_id: str = Field(..., description="Unique prediction identifier")
    message_id: str = Field(..., description="Original message identifier")
    model_version: str = Field(..., description="Model version used")
    timestamp: datetime = Field(..., description="Prediction timestamp")
    extracted_fields: Dict[str, FieldPrediction] = Field(
        ...,
        description="Extracted field predictions"
    )
    overall_confidence: float = Field(
        ...,
        description="Overall prediction confidence"
    )

class FeedbackCorrection(BaseModel):
    """
    Correction for a specific field.
    """
    is_correct: bool = Field(..., description="Whether prediction was correct")
    corrected_value: Optional[str] = Field(
        None,
        description="Correct value if prediction was wrong"
    )

class FeedbackRequest(BaseModel):
    """
    Request schema for prediction feedback.
    """
    prediction_id: str = Field(..., description="Prediction to provide feedback for")
    corrections: Dict[str, FeedbackCorrection] = Field(
        ...,
        description="Corrections for predicted fields"
    )
    notes: Optional[str] = Field(None, description="Additional feedback notes")

class FeedbackResponse(BaseModel):
    """
    Response schema for feedback submission.
    """
    feedback_id: str = Field(..., description="Unique feedback identifier")
    status: str = Field(..., description="Feedback processing status")
    quality_score: float = Field(..., description="Feedback quality score")

    @validator('quality_score')
    def validate_score(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Quality score must be between 0 and 1")
        return v

class ErrorResponse(BaseModel):
    """
    Error response schema.
    """
    status: str = Field("error", description="Error status")
    message: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code")
    details: Optional[Dict] = Field(None, description="Additional error details")

class HealthResponse(BaseModel):
    """
    Health check response schema.
    """
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    model_version: str = Field(..., description="Current model version")
    uptime: float = Field(..., description="Service uptime in seconds")
    components: Dict[str, str] = Field(
        ...,
        description="Status of system components"
    )

class ModelMetrics(BaseModel):
    """
    Model performance metrics schema.
    """
    accuracy: float = Field(..., description="Model accuracy")
    precision: float = Field(..., description="Model precision")
    recall: float = Field(..., description="Model recall")
    f1_score: float = Field(..., description="F1 score")
    confidence: float = Field(..., description="Average confidence")
    sample_count: int = Field(..., description="Number of samples")

class PerformanceResponse(BaseModel):
    """
    Performance metrics response schema.
    """
    model_version: str = Field(..., description="Model version")
    timestamp: datetime = Field(..., description="Metrics timestamp")
    overall_metrics: ModelMetrics = Field(
        ...,
        description="Overall model metrics"
    )
    field_metrics: Dict[str, ModelMetrics] = Field(
        ...,
        description="Field-specific metrics"
    )
    recent_predictions: int = Field(
        ...,
        description="Number of predictions in last hour"
    )
    feedback_rate: float = Field(
        ...,
        description="Rate of feedback received"
    )