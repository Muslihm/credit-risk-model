"""
FastAPI application for credit risk scoring API.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict
import time
from datetime import datetime
import os

from src.api.pydantic_models import (
    CreditApplication, BatchCreditRequest, PredictionResponse,
    BatchPredictionResponse, ModelExplanationResponse, HealthResponse
)
from src.predict import CreditRiskPredictor


# Global predictor instance
predictor = None
MODEL_VERSION = "v1.0.0"
MODEL_PATH = os.getenv("MODEL_PATH", "models/logistic_regression_model.joblib")


def load_model():
    """Load model on startup."""
    global predictor
    try:
        predictor = CreditRiskPredictor(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
        return True
    except Exception as e:
        print(f"Failed to load model: {e}")
        return False


# Create FastAPI app
app = FastAPI(
    title="Credit Risk Scoring API",
    description="API for predicting credit default risk with model explainability",
    version=MODEL_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Load model on application startup."""
    success = load_model()
    if not success:
        print("WARNING: Model failed to load. API will return 503 errors.")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if predictor is not None else "degraded",
        model_loaded=predictor is not None,
        model_type=predictor.model.model_type if predictor else None,
        version=MODEL_VERSION,
        timestamp=datetime.now()
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(application: CreditApplication):
    """
    Predict default probability for a single credit application.
    
    Returns probability of default (0-1) and risk category.
    """
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check server logs."
        )
    
    try:
        # Convert to dict for prediction
        app_dict = application.dict()
        
        # Make prediction
        proba = predictor.predict_proba(app_dict)[0]
        category = predictor.predict_risk_category([proba])[0]
        
        return PredictionResponse(
            default_probability=float(proba),
            risk_category=category,
            model_version=MODEL_VERSION,
            timestamp=datetime.now()
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(batch_request: BatchCreditRequest):
    """
    Predict default probabilities for multiple credit applications.
    
    Maximum 1000 applications per request.
    """
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check server logs."
        )
    
    start_time = time.time()
    
    try:
        # Convert to list of dicts
        applications = [app.dict() for app in batch_request.applications]
        
        # Batch prediction
        probabilities = predictor.predict_proba(applications)
        categories = predictor.predict_risk_category(probabilities)
        
        # Build responses
        predictions = []
        for i, (proba, category) in enumerate(zip(probabilities, categories)):
            predictions.append(PredictionResponse(
                application_id=f"BATCH-{i+1}",
                default_probability=float(proba),
                risk_category=category,
                model_version=MODEL_VERSION,
                timestamp=datetime.now()
            ))
        
        processing_time = (time.time() - start_time) * 1000
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_processed=len(predictions),
            processing_time_ms=round(processing_time, 2)
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.post("/explain", response_model=ModelExplanationResponse)
async def explain_prediction(application: CreditApplication):
    """
    Get explainable prediction for regulatory compliance.
    
    Returns feature contributions and model transparency information.
    """
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check server logs."
        )
    
    try:
        explanation = predictor.explain_prediction(application.dict())
        
        return ModelExplanationResponse(
            default_probability=explanation['default_probability'],
            risk_category=explanation['risk_category'],
            model_type=explanation['model_type'],
            is_interpretable=explanation['is_interpretable'],
            feature_contributions=explanation.get('feature_contributions'),
            intercept=explanation.get('intercept')
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Explanation failed: {str(e)}"
        )


@app.get("/model/info")
async def model_info():
    """Get model metadata and feature information."""
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return {
        "model_type": predictor.model.model_type,
        "version": MODEL_VERSION,
        "selected_features": predictor.model.selected_features,
        "is_interpretable": predictor.model.model_type == 'logistic_regression',
        "has_woe_encoder": predictor.model.woe_encoder is not None,
        "feature_count": len(predictor.model.selected_features)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)