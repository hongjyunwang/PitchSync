import pickle
import pandas as pd
import numpy as np
import os
from typing import Dict, Any, Optional
import logging
from sklearn.preprocessing import LabelEncoder

from pybaseball import playerid_reverse_lookup

logger = logging.getLogger(__name__)

class PitchPredictionService:
    def __init__(self, models_dir: str = "./models/pitcher_models/"):
        self.models_dir = models_dir
        self.loaded_models = {}
        
    def load_pitcher_model(self, pitcher_id: int) -> Optional[Dict[str, Any]]:
        """Load a specific pitcher's model"""
        if pitcher_id in self.loaded_models:
            return self.loaded_models[pitcher_id]
        
        model_path = os.path.join(self.models_dir, f"{pitcher_id}.pkl")
        
        if not os.path.exists(model_path):
            logger.warning(f"No model found for pitcher {pitcher_id}")
            return None
        
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.loaded_models[pitcher_id] = model_data
            logger.info(f"Loaded model for pitcher {pitcher_id}")
            return model_data
        
        except Exception as e:
            logger.error(f"Error loading model for pitcher {pitcher_id}: {e}")
            return None
    
    def get_available_pitchers(self) -> list:
        """Get list of pitcher IDs with trained models"""
        if not os.path.exists(self.models_dir):
            return []
        
        pitcher_files = [f for f in os.listdir(self.models_dir) if f.endswith('.pkl')]
        pitcher_ids = []
        
        for file in pitcher_files:
            try:
                pitcher_id = int(file.replace('.pkl', ''))
                pitcher_ids.append(pitcher_id)
            except ValueError:
                continue
        
        return sorted(pitcher_ids)
    
    def preprocess_input(self, input_data: Dict[str, Any], feature_names: list) -> pd.DataFrame:
        """Preprocess input data to match training format"""
        # Convert to DataFrame if it's a dict
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()
        
        # MATCH TRAINING PREPROCESSING EXACTLY
        
        # Convert 'game_date' to datetime (if present)
        if 'game_date' in df.columns:
            df['game_date'] = pd.to_datetime(df['game_date'])
        
        # Inning breakdown
        if 'inning_topbot' not in df.columns:
            # Assume we're predicting for home team by default
            df['inning_topbot'] = 'Bot'
        df['home_team'] = df['inning_topbot'].map({'Top': 0, 'Bot': 1})
        
        # Count info
        if 'balls' in df.columns and 'strikes' in df.columns:
            df['balls'] = df['balls'].astype(int)
            df['strikes'] = df['strikes'].astype(int)
            df['count'] = df['balls'].astype(str) + '-' + df['strikes'].astype(str)
        
        # Base runner info - encode as boolean (MATCH TRAINING)
        for base in ['on_1b', 'on_2b', 'on_3b']:
            if base in df.columns:
                df[base] = ~pd.isna(df[base]) & (df[base] != 0)
            else:
                df[base] = False
        
        # Outs when up
        if 'outs_when_up' in df.columns:
            df['outs_when_up'] = df['outs_when_up'].astype(int)
        else:
            df['outs_when_up'] = 0
        
        # Score differential (MATCH TRAINING LOGIC)
        if 'fld_score' in df.columns and 'bat_score' in df.columns:
            # Assume we're predicting for home team (Bot inning)
            if df['inning_topbot'].iloc[0] == 'Bot':
                df['score_diff'] = df['fld_score'] - df['bat_score']  # Home team perspective
            else:
                df['score_diff'] = df['bat_score'] - df['fld_score']  # Away team perspective
        else:
            df['score_diff'] = 0
        
        # Game state features (MATCH TRAINING)
        if 'inning' in df.columns:
            df['late_game'] = (df['inning'] >= 7).astype(int)
        else:
            df['inning'] = 1
            df['late_game'] = 0
        
        df['runners_on'] = (df['on_1b'] | df['on_2b'] | df['on_3b']).astype(int)
        df['scoring_position'] = (df['on_2b'] | df['on_3b']).astype(int)
        df['two_out_scoring'] = ((df['outs_when_up'] == 2) & df['scoring_position']).astype(int)
        
        # Pitch number within at-bat (if not provided, assume 1)
        if 'pitch_number_at_bat' not in df.columns:
            df['pitch_number_at_bat'] = 1
        
        # Encode categorical variables (MATCH TRAINING)
        categorical_columns = ['count', 'stand', 'p_throws']
        
        for col in categorical_columns:
            if col in df.columns:
                # Simple label encoding (same as training)
                le = LabelEncoder()
                unique_values = df[col].astype(str).unique()
                le.fit(unique_values)
                df[col + '_encoded'] = le.transform(df[col].astype(str))
            else:
                # Default values if missing
                if col == 'count':
                    df[col] = '0-0'
                    df[col + '_encoded'] = 0
                elif col == 'stand':
                    df[col] = 'R'
                    df[col + '_encoded'] = 0
                elif col == 'p_throws':
                    df[col] = 'R'
                    df[col + '_encoded'] = 0
        
        # Create all required features with defaults
        required_features = [
            'inning', 'home_team', 'balls', 'strikes', 'on_3b', 'on_2b', 'on_1b',
            'outs_when_up', 'score_diff', 'late_game', 'runners_on', 'scoring_position',
            'two_out_scoring', 'pitch_number_at_bat', 'count_encoded', 'stand_encoded', 'p_throws_encoded'
        ]
        
        for feature in required_features:
            if feature not in df.columns:
                df[feature] = 0  # Default value
        
        # Select only the features that were used in training
        available_features = [col for col in feature_names if col in df.columns]
        missing_features = [col for col in feature_names if col not in df.columns]
        
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            # Add missing features with default values
            for feature in missing_features:
                df[feature] = 0
        
        # Return only the features used in training, in the same order
        return df[feature_names]
    
    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction for a single pitch scenario - FIXED for new training approach"""
        try:
            # Extract pitcher ID
            pitcher_id = input_data.get('pitcher')
            if not pitcher_id:
                return {"error": "Pitcher ID is required"}
            
            # Load model
            model_data = self.load_pitcher_model(pitcher_id)
            if not model_data:
                return {"error": f"No model available for pitcher {pitcher_id}"}
            
            model = model_data['model']
            label_encoder = model_data['label_encoder']  # NEW: Get the encoder
            feature_names = model_data['feature_names']
            
            # Preprocess input to match training format
            processed_df = self.preprocess_input(input_data, feature_names)
            
            # Make prediction - FIXED for new encoding approach
            prediction_encoded = model.predict(processed_df)[0]  # Encoded prediction (integer)
            probabilities = model.predict_proba(processed_df)[0]  # All probabilities
            
            # Decode prediction back to string
            prediction = label_encoder.inverse_transform([prediction_encoded])[0]
            
            # Get class labels (decode all classes)
            classes_encoded = model.classes_  # [0, 1, 2, 3, ...]
            classes = label_encoder.inverse_transform(classes_encoded)  # ['Fastball', 'Slider', ...]
            
            # Create probability dictionary
            pitch_probabilities = {}
            for i, pitch_type in enumerate(classes):
                pitch_probabilities[pitch_type] = float(probabilities[i])
            
            # Sort probabilities by likelihood
            sorted_probabilities = dict(sorted(pitch_probabilities.items(), 
                                             key=lambda x: x[1], reverse=True))
            
            # Enhanced result format
            result = {
                "pitcher_id": pitcher_id,
                "prediction": int(prediction in ['Fastball', 'FF', 'FA', 'FT', 'SI']),  # Backward compatibility
                "is_fastball": bool(prediction in ['Fastball', 'FF', 'FA', 'FT', 'SI']),  # Backward compatibility
                "predicted_pitch": str(prediction),  # Specific pitch type
                "confidence": float(max(probabilities)),
                "pitch_probabilities": sorted_probabilities,  # All probabilities
                "pitch_arsenal": model_data.get('pitch_arsenal', {}),  # Pitcher's arsenal
                "filtered_pitches": model_data.get('filtered_pitches', []),  # NEW: Filtered out pitches
                "top_3_predictions": dict(list(sorted_probabilities.items())[:3]),  # Top 3
                "model_accuracy": model_data.get('model_accuracy', 'N/A'),
                "total_pitch_types": len(classes),  # Number of pitch types
                "validation_method": model_data.get('validation_method', 'N/A'),  # NEW
                
                # Backward compatibility fields
                "probability_fastball": float(sum(pitch_probabilities.get(p, 0.0) 
                                                for p in ['Fastball', 'FF', 'FA', 'FT', 'SI'])),
                "probability_offspeed": float(1.0 - sum(pitch_probabilities.get(p, 0.0) 
                                                       for p in ['Fastball', 'FF', 'FA', 'FT', 'SI']))
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return {"error": f"Prediction failed: {str(e)}"}
    
    def get_pitcher_info(self, pitcher_id: int) -> Dict[str, Any]:
        """Get information about a specific pitcher's model - ENHANCED"""
        model_data = self.load_pitcher_model(pitcher_id)
        if not model_data:
            return {"error": f"No model available for pitcher {pitcher_id}"}
        
        try:
            # Get name and additional info
            player_info_df = playerid_reverse_lookup([pitcher_id])
            if not player_info_df.empty:
                name = player_info_df.iloc[0]['name_first'] + " " + player_info_df.iloc[0]['name_last']
            else:
                name = "Unknown"
        except Exception as e:
            logger.warning(f"Could not retrieve player name for ID {pitcher_id}: {e}")
            name = "Unknown"
        
        # Enhanced pitcher info (compatible with new training)
        return {
            "pitcher_name": name,
            "pitcher_id": pitcher_id,
            "model_accuracy": model_data.get('model_accuracy', 'N/A'),
            "naive_accuracy": model_data.get('naive_accuracy', 'N/A'),
            "training_samples": model_data.get('training_samples', 'N/A'),
            "test_samples": model_data.get('test_samples', 'N/A'),
            "best_params": model_data.get('best_params', {}),
            "pitch_arsenal": model_data.get('pitch_arsenal', {}),
            "pitch_types": model_data.get('pitch_types', []),
            "filtered_pitches": model_data.get('filtered_pitches', []),  # NEW
            "rare_pitch_threshold": model_data.get('rare_pitch_threshold', 5),  # NEW
            "validation_method": model_data.get('validation_method', 'N/A'),  # NEW
            "classification_report": model_data.get('classification_report', {})
        }
