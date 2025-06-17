from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import uvicorn

from prediction_service import PitchPredictionService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="MLB Pitch Prediction API",
    description="API for predicting MLB pitcher's next pitch type",
    version="2.0.0"  # Updated version for new training approach
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize prediction service
prediction_service = PitchPredictionService()

# Pydantic models for request/response validation
class PitchScenario(BaseModel):
    """Input model for pitch prediction"""
    pitcher: int = Field(..., description="Pitcher ID")
    inning: int = Field(..., ge=1, le=20, description="Current inning") 
    balls: int = Field(..., ge=0, le=3, description="Ball count")
    strikes: int = Field(..., ge=0, le=2, description="Strike count")
    outs_when_up: int = Field(..., ge=0, le=2, description="Number of outs")
    
    # Base runners (optional)
    on_1b: Optional[float] = Field(None, description="Runner on 1st base (player ID or NaN)")
    on_2b: Optional[float] = Field(None, description="Runner on 2nd base (player ID or NaN)")
    on_3b: Optional[float] = Field(None, description="Runner on 3rd base (player ID or NaN)")
    
    # Score (optional)
    fld_score: Optional[int] = Field(0, description="Fielding team score")
    bat_score: Optional[int] = Field(0, description="Batting team score")
    
    # Batter info (optional)
    batter: Optional[int] = Field(12345, description="Batter ID")
    stand: Optional[str] = Field("R", description="Batter stance (L/R)")
    
    # Pitcher info (optional)
    p_throws: Optional[str] = Field("R", description="Pitcher throws (L/R)")
    
    # Game info (optional)
    game_pk: Optional[int] = Field(12345, description="Game ID")
    pitch_number_at_bat: Optional[int] = Field(1, description="Pitch number in at-bat")
    inning_topbot: Optional[str] = Field("Bot", description="Top or Bot of inning")

class PredictionResponse(BaseModel):
    """Enhanced response model for pitch prediction"""
    pitcher_id: int
    prediction: int  # Backward compatibility (1 for fastball, 0 for offspeed)
    is_fastball: bool  # Backward compatibility
    predicted_pitch: str  # NEW: Specific pitch type
    confidence: float
    pitch_probabilities: Dict[str, float]  # NEW: All pitch probabilities
    pitch_arsenal: Dict[str, int]  # NEW: Pitcher's arsenal
    filtered_pitches: List[str]  # NEW: Pitch types filtered out during training
    top_3_predictions: Dict[str, float]  # NEW: Top 3 predictions
    model_accuracy: float
    total_pitch_types: int  # NEW: Number of pitch types
    validation_method: str  # NEW: Validation method used during training
    probability_fastball: float  # Backward compatibility
    probability_offspeed: float  # Backward compatibility
    timestamp: datetime = Field(default_factory=datetime.now)

class PitcherInfo(BaseModel):
    """Enhanced response model for pitcher information"""
    pitcher_name: str
    pitcher_id: int
    model_accuracy: float
    naive_accuracy: float
    training_samples: int
    test_samples: int
    best_params: Dict[str, Any]
    pitch_arsenal: Dict[str, int]  # Pitcher's arsenal
    pitch_types: List[str]  # List of pitch types
    filtered_pitches: List[str]  # NEW: Pitch types filtered during training
    rare_pitch_threshold: int  # NEW: Threshold used for filtering
    validation_method: str  # NEW: Validation method used
    classification_report: Dict[str, Any]  # Detailed classification metrics

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    timestamp: datetime = Field(default_factory=datetime.now)

# API Routes
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "MLB Pitch Prediction API",
        "version": "2.0.0",
        "features": [
            "Binary classification (fastball vs offspeed) - LEGACY",
            "Multiclass classification (specific pitch types) - NEW", 
            "Probability distribution for all pitch types - NEW",
            "Pitcher arsenal analysis - NEW",
            "Rare pitch filtering during training - NEW",
            "Enhanced validation strategies - NEW"
        ],
        "endpoints": {
            "predict": "/predict",
            "pitchers": "/pitchers",
            "pitcher_info": "/pitcher/{pitcher_id}",
            "pitcher_arsenal": "/pitcher/{pitcher_id}/arsenal",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "available_pitchers": len(prediction_service.get_available_pitchers())
    }

@app.get("/pitchers", response_model=List[int])
async def get_available_pitchers():
    """Get list of pitcher IDs with trained models"""
    try:
        pitchers = prediction_service.get_available_pitchers()
        return pitchers
    except Exception as e:
        logger.error(f"Error getting pitchers: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving pitcher list")

@app.get("/pitcher/{pitcher_id}", response_model=PitcherInfo)
async def get_pitcher_info(pitcher_id: int):
    """Get information about a specific pitcher's model"""
    try:
        info = prediction_service.get_pitcher_info(pitcher_id)
        if "error" in info:
            raise HTTPException(status_code=404, detail=info["error"])
        return info
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting pitcher info: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving pitcher information")

@app.post("/predict", response_model=PredictionResponse)
async def predict_pitch(scenario: PitchScenario):
    """Predict the next pitch type for a given scenario"""
    try:
        # Convert Pydantic model to dict
        input_data = scenario.dict()
        
        # Make prediction
        result = prediction_service.predict(input_data)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.post("/predict/batch")
async def predict_batch(scenarios: List[PitchScenario]):
    """Predict pitch types for multiple scenarios"""
    if len(scenarios) > 100:  # Limit batch size
        raise HTTPException(status_code=400, detail="Batch size cannot exceed 100")
    
    results = []
    errors = []
    
    for i, scenario in enumerate(scenarios):
        try:
            input_data = scenario.dict()
            result = prediction_service.predict(input_data)
            if "error" in result:
                errors.append({"index": i, "error": result["error"]})
            else:
                results.append(result)
        except Exception as e:
            errors.append({"index": i, "error": str(e)})
    
    return {
        "results": results,
        "errors": errors,
        "total_processed": len(scenarios),
        "successful": len(results),
        "failed": len(errors)
    }

# Enhanced endpoints
@app.get("/pitcher/{pitcher_id}/arsenal")
async def get_pitcher_arsenal(pitcher_id: int):
    """Get detailed information about a pitcher's arsenal"""
    try:
        info = prediction_service.get_pitcher_info(pitcher_id)
        if "error" in info:
            raise HTTPException(status_code=404, detail=info["error"])
        
        arsenal = info.get("pitch_arsenal", {})
        total_pitches = sum(arsenal.values()) if arsenal else 0
        filtered_pitches = info.get("filtered_pitches", [])
        
        # Calculate percentages
        arsenal_percentages = {}
        if total_pitches > 0:
            for pitch_type, count in arsenal.items():
                arsenal_percentages[pitch_type] = {
                    "count": count,
                    "percentage": round((count / total_pitches) * 100, 2)
                }
        
        return {
            "pitcher_id": pitcher_id,
            "pitcher_name": info.get("pitcher_name", "Unknown"),
            "total_pitches": total_pitches,
            "arsenal": arsenal_percentages,
            "pitch_types": list(arsenal.keys()),
            "filtered_pitches": filtered_pitches,  # NEW: Show what was filtered
            "rare_pitch_threshold": info.get("rare_pitch_threshold", 5),  # NEW
            "primary_pitch": max(arsenal, key=arsenal.get) if arsenal else "Unknown",
            "validation_method": info.get("validation_method", "N/A")  # NEW
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting pitcher arsenal: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving pitcher arsenal")

@app.get("/pitcher/{pitcher_id}/training-details")
async def get_pitcher_training_details(pitcher_id: int):
    """Get detailed training information for a pitcher"""
    try:
        info = prediction_service.get_pitcher_info(pitcher_id)
        if "error" in info:
            raise HTTPException(status_code=404, detail=info["error"])
        
        return {
            "pitcher_id": pitcher_id,
            "pitcher_name": info.get("pitcher_name", "Unknown"),
            "training_details": {
                "model_accuracy": info.get("model_accuracy", "N/A"),
                "naive_accuracy": info.get("naive_accuracy", "N/A"),
                "improvement": round(float(info.get("model_accuracy", 0)) - float(info.get("naive_accuracy", 0)), 2) if info.get("model_accuracy", "N/A") != "N/A" else "N/A",
                "training_samples": info.get("training_samples", "N/A"),
                "test_samples": info.get("test_samples", "N/A"),
                "validation_method": info.get("validation_method", "N/A"),
                "best_params": info.get("best_params", {}),
                "rare_pitch_threshold": info.get("rare_pitch_threshold", 5),
                "filtered_pitches": info.get("filtered_pitches", [])
            },
            "classification_report": info.get("classification_report", {})
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting training details: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving training details")

# Example usage endpoint
@app.get("/example")
async def get_example_request():
    """Get an example request for testing"""
    return {
        "example_request": {
            "pitcher": 621111,  # Use a pitcher ID from your trained models
            "inning": 3,
            "balls": 2,
            "strikes": 1,
            "outs_when_up": 1,
            "on_1b": None,
            "on_2b": 12345,
            "on_3b": None,
            "fld_score": 2,
            "bat_score": 1,
            "stand": "L",
            "p_throws": "R",
            "inning_topbot": "Bot"
        },
        "enhanced_response_fields": {
            "predicted_pitch": "Specific pitch type (FF, SL, CH, etc.)",
            "pitch_probabilities": "Probabilities for all pitch types",
            "pitch_arsenal": "Pitcher's complete arsenal with counts",
            "filtered_pitches": "Pitch types filtered during training",
            "validation_method": "Training validation approach used",
            "top_3_predictions": "Top 3 most likely pitches"
        },
        "legacy_compatibility": {
            "prediction": "1 for fastball, 0 for offspeed",
            "is_fastball": "boolean",
            "probability_fastball": "combined fastball probability",
            "probability_offspeed": "combined offspeed probability"
        },
        "curl_example": """
curl -X POST "http://localhost:8000/predict" \\
     -H "Content-Type: application/json" \\
     -d '{
       "pitcher": 621111,
       "inning": 3,
       "balls": 2,
       "strikes": 1,
       "outs_when_up": 1,
       "on_2b": 12345,
       "fld_score": 2,
       "bat_score": 1,
       "stand": "L",
       "p_throws": "R",
       "inning_topbot": "Bot"
     }'
        """
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload during development
        log_level="info"
    )
    