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
    version="1.0.0"
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
    
    # Previous pitch info (optional)
    prev_type: Optional[str] = Field("UN", description="Previous pitch result")
    prev_pfx_x: Optional[float] = Field(0.0, description="Previous pitch horizontal movement")
    prev_pfx_z: Optional[float] = Field(0.0, description="Previous pitch vertical movement")
    prev_plate_x: Optional[float] = Field(0.0, description="Previous pitch plate X location")
    prev_plate_z: Optional[float] = Field(0.0, description="Previous pitch plate Z location")
    prev_release_speed: Optional[float] = Field(0.0, description="Previous pitch velocity")
    prev_release_spin_rate: Optional[float] = Field(0.0, description="Previous pitch spin rate")
    prev_pitch_type: Optional[str] = Field("UN", description="Previous pitch type")
    
    # Game info (optional)
    game_pk: Optional[int] = Field(12345, description="Game ID")
    pitch_number: Optional[int] = Field(1, description="Pitch number in at-bat")

class PredictionResponse(BaseModel):
    """Response model for pitch prediction"""
    pitcher_id: int
    prediction: int
    is_fastball: bool
    probability_fastball: float
    probability_offspeed: float
    model_accuracy: float
    confidence: float
    timestamp: datetime = Field(default_factory=datetime.now)

class PitcherInfo(BaseModel):
    """Response model for pitcher information"""
    pitcher_id: int
    model_accuracy: float
    naive_accuracy: float
    training_samples: int
    test_samples: int
    best_params: Dict[str, Any]

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
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "pitchers": "/pitchers",
            "pitcher_info": "/pitcher/{pitcher_id}",
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

# Example usage endpoint
@app.get("/example")
async def get_example_request():
    """Get an example request for testing"""
    return {
        "example_request": {
            "pitcher": 543037,  # Example pitcher ID
            "inning": 3,
            "balls": 2,
            "strikes": 1,
            "outs_when_up": 1,
            "on_1b": None,
            "on_2b": 12345,
            "on_3b": None,
            "fld_score": 2,
            "bat_score": 1,
            "batter": 67890,
            "stand": "L",
            "p_throws": "R"
        },
        "curl_example": """
curl -X POST "http://localhost:8000/predict" \\
     -H "Content-Type: application/json" \\
     -d '{
       "pitcher": 543037,
       "inning": 3,
       "balls": 2,
       "strikes": 1,
       "outs_when_up": 1,
       "on_2b": 12345,
       "fld_score": 2,
       "bat_score": 1,
       "stand": "L",
       "p_throws": "R"
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