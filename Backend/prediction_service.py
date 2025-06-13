import pickle
import pandas as pd
import numpy as np
import os
from typing import Dict, Any, Optional
import logging

from pybaseball import playerid_reverse_lookup

logger = logging.getLogger(__name__)

class PitchPredictionService:
    def __init__(self, models_dir: str = "./models/pitcher_models/"):
        self.models_dir = models_dir
        self.loaded_models = {}
        self.fastball_pitches = ['FA', 'FF', 'FT', 'FC', 'FS', 'SI', 'SF']
        
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
    
    def preprocess_input(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """Preprocess input data to match training format"""
        # Convert to DataFrame if it's a dict
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()
        
        # Map fastball pitches if pitch_type is present
        if 'pitch_type' in df.columns:
            df['pitch_type'] = df['pitch_type'].apply(self._map_fastballs)
        
        # Basic type casting
        int_columns = ['game_pk', 'pitcher', 'batter', 'inning', 'balls', 'strikes', 
                      'outs_when_up', 'pitch_number']
        for col in int_columns:
            if col in df.columns:
                df[col] = df[col].astype('int64')
        
        # Cap innings at 9
        if 'inning' in df.columns:
            df['inning'] = df['inning'].apply(lambda x: 9 if x > 9 else x)
        
        # Create game_pitcher_id
        if 'game_pk' in df.columns and 'pitcher' in df.columns:
            df['game_pitcher_id'] = df['game_pk'].astype(str) + '_' + df['pitcher'].astype(str)
        
        # Handle base runners
        for base in ['on_1b', 'on_2b', 'on_3b']:
            if base in df.columns:
                df[base] = df[base].apply(lambda x: not pd.isna(x) and x != 0)
        
        # Handle handedness
        if 'p_throws' in df.columns and 'stand' in df.columns:
            df['pitch_bat_same_side'] = df['p_throws'] == df['stand']
            df.drop(['p_throws', 'stand'], axis=1, inplace=True)
        
        # Score differential
        if 'fld_score' in df.columns and 'bat_score' in df.columns:
            df['score_diff'] = df['fld_score'] - df['bat_score']
            df.drop(['fld_score', 'bat_score'], axis=1, inplace=True)
        
        # Handle previous pitch features
        prev_pitch_cols = ['prev_type', 'prev_pfx_x', 'prev_pfx_z', 'prev_plate_x', 
                          'prev_plate_z', 'prev_release_speed', 'prev_release_spin_rate', 
                          'prev_pitch_type']
        
        for col in prev_pitch_cols:
            if col not in df.columns:
                if col in ['prev_type', 'prev_pitch_type']:
                    df[col] = 'UN'  # Unknown for categorical
                else:
                    df[col] = 0.0   # Zero for numerical
        
        # Fill missing values
        for col in prev_pitch_cols:
            if col in ['prev_type', 'prev_pitch_type']:
                df[col].fillna('UN', inplace=True)
            else:
                df[col].fillna(0.0, inplace=True)
        
        # Add default fb_prob (ideally this would come from pre-computed lookup)
        if 'fb_prob' not in df.columns:
            df['fb_prob'] = 0.5
        
        # One-hot encoding
        cols_to_encode = ['inning', 'balls', 'strikes', 'outs_when_up', 'prev_type', 'prev_pitch_type']
        
        for col in cols_to_encode:
            if col in df.columns:
                # Get dummies and add to dataframe
                dummies = pd.get_dummies(df[col], prefix=col)
                df = pd.concat([df, dummies], axis=1)
                df.drop(col, axis=1, inplace=True)
        
        # Drop ID columns
        id_cols = ['game_pk', 'batter', 'game_pitcher_id']
        for col in id_cols:
            if col in df.columns:
                df.drop(col, axis=1, inplace=True)
        
        return df
    
    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction for a single pitch scenario"""
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
            feature_names = model_data['feature_names']
            
            # Preprocess input
            processed_df = self.preprocess_input(input_data)
            
            # Ensure all required features are present
            for feature in feature_names:
                if feature not in processed_df.columns:
                    processed_df[feature] = 0
            
            # Select only the features used in training
            processed_df = processed_df[feature_names]
            
            # Make prediction
            prediction = model.predict(processed_df)[0] # Fastball or not
            probabilities = model.predict_proba(processed_df)[0] # Specific probability of fastball or offspeed
            
            # Format results
            result = {
                "pitcher_id": pitcher_id,
                "prediction": int(prediction),
                "is_fastball": bool(prediction == 1),
                "probability_fastball": float(probabilities[1]) if len(probabilities) > 1 else float(probabilities[0]),
                "probability_offspeed": float(probabilities[0]) if len(probabilities) > 1 else float(1 - probabilities[0]),
                "model_accuracy": model_data.get('model_accuracy', 'N/A'),
                "confidence": float(max(probabilities))
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return {"error": f"Prediction failed: {str(e)}"}
    
    def get_pitcher_info(self, pitcher_id: int) -> Dict[str, Any]:
        """Get information about a specific pitcher's model"""
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
        
        return {
            "pitcher_name": name,
            "pitcher_id": pitcher_id,
            "model_accuracy": model_data.get('model_accuracy', 'N/A'),
            "naive_accuracy": model_data.get('naive_accuracy', 'N/A'),
            "training_samples": model_data.get('training_samples', 'N/A'),
            "test_samples": model_data.get('test_samples', 'N/A'),
            "best_params": model_data.get('best_params', {})
        }
    
    def _map_fastballs(self, pitch_type):
        """Map pitch types to fastball (1) or off-speed (0)"""
        return 1 if pitch_type in self.fastball_pitches else 0
    