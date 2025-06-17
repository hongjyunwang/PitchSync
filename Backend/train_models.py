import re
import pickle
import datetime as dt
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import os
from pybaseball import statcast



class PitchTrainer:
    """Train individual pitch prediction models for each pitcher"""
    
    def __init__(self):
        # Create directories if they don't exist
        os.makedirs('./data/', exist_ok=True)
        os.makedirs('./models/', exist_ok=True)
        os.makedirs('./Backend/models/pitcher_models/', exist_ok=True)
    
    def extract_features(self, data):
        """Extract features from the statcast data"""
        print("Engineering features...")
        
        # Convert 'game_date' to datetime object if it's not already
        data['game_date'] = pd.to_datetime(data['game_date'])
        
        # Inning breakdown
        data['home_team'] = data['inning_topbot'].map({'Top': 0, 'Bot': 1})
        
        # Count info
        data['balls'] = data['balls'].astype(int)
        data['strikes'] = data['strikes'].astype(int)
        data['count'] = data['balls'].astype(str) + '-' + data['strikes'].astype(str)
        
        # Base runner info - encode as boolean
        data['on_3b'] = ~data['on_3b'].isna()
        data['on_2b'] = ~data['on_2b'].isna()
        data['on_1b'] = ~data['on_1b'].isna()
        
        # Outs when up
        data['outs_when_up'] = data['outs_when_up'].astype(int)
        
        # Score differential (positive = pitcher's team winning)
        # When home team is pitching (Bot), use home_score - away_score
        # When away team is pitching (Top), use away_score - home_score
        data['score_diff'] = np.where(
            data['inning_topbot'] == 'Bot',
            data['home_score'] - data['fld_score'],
            data['fld_score'] - data['home_score']
        )
        
        # Game state features
        data['late_game'] = (data['inning'] >= 7).astype(int)
        data['runners_on'] = (data['on_1b'] | data['on_2b'] | data['on_3b']).astype(int)
        data['scoring_position'] = (data['on_2b'] | data['on_3b']).astype(int)
        data['two_out_scoring'] = ((data['outs_when_up'] == 2) & data['scoring_position']).astype(int)
        
        # Calculate pitch number within at-bat (could be used as fatigue indicator)
        data['pitch_number_at_bat'] = data.groupby(['game_pk', 'at_bat_number']).cumcount() + 1
        
        # Encode categorical variables as numeric
        categorical_columns = ['count', 'stand', 'p_throws']
        
        for col in categorical_columns:
            if col in data.columns:
                # Use label encoding for categorical variables
                le = LabelEncoder()
                data[col + '_encoded'] = le.fit_transform(data[col].astype(str))
        
        # Select relevant features for model
        feature_columns = [
            'pitcher',
            'pitch_type',
            'inning',
            'home_team',
            'balls',
            'strikes',
            'on_3b',
            'on_2b', 
            'on_1b',
            'outs_when_up',
            'score_diff',
            'late_game',
            'runners_on',
            'scoring_position',
            'two_out_scoring',
            'pitch_number_at_bat',
            'count_encoded',
            'stand_encoded',
            'p_throws_encoded'
        ]
        
        # Filter to only include columns that exist in the data
        available_columns = [col for col in feature_columns if col in data.columns]
        
        print(f"Selected {len(available_columns)} features: {available_columns}")
        
        return data[available_columns]
    
    def load_and_clean_data(self, start_date='2024-03-20', end_date='2024-11-02'):
        """Load and clean data from the specified date range"""
        print(f"Loading data from {start_date} to {end_date}")
        
        # Download data
        raw_data = statcast(start_dt=start_date, end_dt=end_date)
        print(f"Raw data shape: {raw_data.shape}")
        
        # Remove rows with null pitch types
        clean_data = raw_data.dropna(subset=['pitch_type'])
        print(f"After dropping null pitch types: {clean_data.shape}")
        
        # Filter out rare pitch types (less than 500 occurrences)
        pitch_counts = clean_data['pitch_type'].value_counts()
        common_pitches = pitch_counts[pitch_counts >= 500].index
        clean_data = clean_data[clean_data['pitch_type'].isin(common_pitches)]
        print(f"After filtering rare pitch types: {clean_data.shape}")
        print(f"Remaining pitch types: {sorted(clean_data['pitch_type'].unique())}")
        
        return clean_data
    
    def train_models(self, data, pitch_count_cutoff=1000, rare_pitch_threshold=5):
        """Train individual models for each pitcher, filtering out rare pitches per pitcher"""
        # Ensure directory exists
        os.makedirs('./Backend/models/pitcher_models/', exist_ok=True)
        
        # Import required modules
        from sklearn.model_selection import StratifiedKFold
        
        # Get pitcher counts and filter out pitchers with too few pitches
        pitcher_count_dict = dict(Counter(data['pitcher']))
        pitcher_count_dict = {k: v for k, v in pitcher_count_dict.items() if v > pitch_count_cutoff}
        
        print(f"Training models for {len(pitcher_count_dict)} pitchers")
        print(f"Filtering pitch types with < {rare_pitch_threshold} occurrences per pitcher")
        
        accuracy_list = []
        naive_accuracy_list = []
        
        for i, pitcher in enumerate(pitcher_count_dict.keys()):
            start = dt.datetime.now()
            
            # Get pitcher data
            df_pitcher = data[data['pitcher'] == pitcher].copy()
            
            # Get pitch type counts for this pitcher
            pitch_type_counts = df_pitcher['pitch_type'].value_counts()
            
            # Filter out rare pitch types for this pitcher
            common_pitches = pitch_type_counts[pitch_type_counts >= rare_pitch_threshold].index
            df_pitcher_filtered = df_pitcher[df_pitcher['pitch_type'].isin(common_pitches)].copy()
            
            # Check if pitcher has enough variety after filtering
            remaining_pitch_counts = df_pitcher_filtered['pitch_type'].value_counts()
            if len(remaining_pitch_counts) < 2:
                if len(remaining_pitch_counts) == 1:
                    print(f"Skipping pitcher {pitcher}: Only {remaining_pitch_counts.index[0]} after filtering rare pitches")
                else:
                    print(f"Skipping pitcher {pitcher}: No pitches remaining after filtering")
                continue
            
            # Show what was filtered
            filtered_pitches = set(pitch_type_counts.index) - set(common_pitches)
            if filtered_pitches:
                filtered_info = {pitch: pitch_type_counts[pitch] for pitch in filtered_pitches}
                print(f"Pitcher {pitcher}: Filtered rare pitches: {filtered_info}")
            
            print(f"Pitcher {pitcher}: Training on {len(common_pitches)} pitch types: {remaining_pitch_counts.to_dict()}")
            
            # Use the filtered data
            df_pitcher_filtered.drop('pitcher', axis=1, inplace=True)
            
            # Split features and target
            X = df_pitcher_filtered.drop('pitch_type', axis=1)
            y = df_pitcher_filtered['pitch_type']
            
            # Encode string labels to integers for XGBoost
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            
            # Train/test split - always use stratified split
            try:
                X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(
                    X, y_encoded, test_size=0.2, random_state=4256, stratify=y_encoded
                )
                split_method = "stratified"
            except ValueError as e:
                print(f"Skipping pitcher {pitcher}: Stratified split failed - {str(e)[:50]}...")
                continue
            
            # Check if both train and test sets have multiple classes
            train_classes = len(np.unique(y_train_encoded))
            test_classes = len(np.unique(y_test_encoded))
            
            if train_classes < 2 or test_classes < 2:
                print(f"Skipping pitcher {pitcher}: Insufficient class variety after split (train: {train_classes}, test: {test_classes})")
                print(f"Train classes: {np.unique(y_train_encoded)}, Test classes: {np.unique(y_test_encoded)}")
                continue
            
            # Debug info for problematic cases
            if len(remaining_pitch_counts) == 2:
                print(f"Pitcher {pitcher}: Binary classification - Train classes: {np.unique(y_train_encoded)}, Test classes: {np.unique(y_test_encoded)}")
            
            # Determine validation strategy based on remaining data
            min_count = remaining_pitch_counts.min()
            total_samples = len(y_train_encoded)
            
            # More conservative validation strategy to avoid CV failures
            if min_count >= 15 and total_samples >= 100:
                # Use cross-validation only when we have sufficient samples per class
                cv_folds = 5
                use_cv = True
            elif min_count >= 10 and total_samples >= 60:
                # Use fewer CV folds for moderate datasets
                cv_folds = 3
                use_cv = True
            else:
                # Use default parameters for small/imbalanced datasets
                use_cv = False
            
            # XGBoost hyperparameter grid
            xgb_params = {
                "max_depth": [2, 5, 20],
                "learning_rate": [0.01, 0.1, 0.4]
            }
            
            # Create mock GridSearchCV object for consistency
            class MockGridSearchCV:
                def __init__(self, best_estimator, best_params):
                    self.best_estimator_ = best_estimator
                    self.best_params_ = best_params
            
            if use_cv:
                # Try cross-validation with explicit stratification
                try:
                    # Use StratifiedKFold to ensure stratification
                    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                    
                    xgb_opt = GridSearchCV(
                        XGBClassifier(
                            random_state=42,
                            objective='multi:softprob',
                            eval_metric='mlogloss'
                        ), 
                        param_grid=xgb_params, 
                        cv=skf,  # Use explicit StratifiedKFold
                        scoring='accuracy', 
                        verbose=0, 
                        n_jobs=-1
                    )
                    xgb_opt.fit(X_train, y_train_encoded)
                    validation_method = f"{cv_folds}-fold Stratified CV"
                
                except Exception as e:
                    # Fallback to default parameters if CV fails
                    print(f"Pitcher {pitcher}: Stratified CV failed ({str(e)[:50]}...), using default params")
                    try:
                        xgb_model = XGBClassifier(
                            random_state=42,
                            objective='multi:softprob',
                            eval_metric='mlogloss'
                        )
                        xgb_model.fit(X_train, y_train_encoded)
                        xgb_opt = MockGridSearchCV(xgb_model, {})
                        validation_method = "default params (Stratified CV failed)"
                    except Exception as e2:
                        print(f"Skipping pitcher {pitcher}: All training failed ({str(e2)[:50]}...)")
                        continue
            else:
                # Use default parameters for small datasets
                try:
                    xgb_model = XGBClassifier(
                        random_state=42,
                        objective='multi:softprob',
                        eval_metric='mlogloss'
                    )
                    xgb_model.fit(X_train, y_train_encoded)
                    xgb_opt = MockGridSearchCV(xgb_model, {})
                    validation_method = "default params"
                except Exception as e:
                    print(f"Skipping pitcher {pitcher}: XGBoost training failed ({str(e)[:50]}...)")
                    continue
            
            # Make predictions
            y_pred_encoded = xgb_opt.best_estimator_.predict(X_test)
            
            # Decode predictions back to string labels
            y_pred = label_encoder.inverse_transform(y_pred_encoded)
            y_test = label_encoder.inverse_transform(y_test_encoded)
            
            # Calculate accuracies
            accuracy = round(accuracy_score(y_test, y_pred) * 100, 1)
            naive_accuracy = round((pd.Series(y_test).value_counts().iloc[0] / len(y_test)) * 100, 1)
            
            accuracy_list.append(accuracy)
            naive_accuracy_list.append(naive_accuracy)
            
            # Get pitch arsenal (use filtered y for correct counts)
            pitch_arsenal = y.value_counts().to_dict()
            
            # Save model with metadata
            pitcher_model = {
                'model': xgb_opt.best_estimator_,
                'label_encoder': label_encoder,
                'feature_names': X.columns.tolist(),
                'pitch_arsenal': pitch_arsenal,
                'pitch_types': list(pitch_arsenal.keys()),
                'filtered_pitches': list(filtered_pitches) if filtered_pitches else [],
                'rare_pitch_threshold': rare_pitch_threshold,
                'model_accuracy': accuracy,
                'naive_accuracy': naive_accuracy,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'best_params': xgb_opt.best_params_,
                'validation_method': validation_method,
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
            
            model_path = f'./Backend/models/pitcher_models/{pitcher}.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(pitcher_model, f)
            
            if i % 10 == 0:
                print(f"Pitcher {pitcher}: {accuracy}% accuracy (vs {naive_accuracy:.1f}% naive) [{validation_method}]")
                print(f"Arsenal: {list(pitch_arsenal.keys())}")
                if filtered_pitches:
                    print(f"Filtered: {list(filtered_pitches)}")
                print(f"Training time: {dt.datetime.now() - start}")
        
        return accuracy_list, naive_accuracy_list

def main():
    # Initialize trainer
    trainer = PitchTrainer()
    
    # Load and process data
    raw_data = trainer.load_and_clean_data()
    clean_data = trainer.extract_features(raw_data)
    
    # Train models
    accuracy_list, naive_accuracy_list = trainer.train_models(clean_data)
    
    # Print summary statistics
    print(f"\n--- Training Complete ---")
    print(f"Models trained: {len(accuracy_list)}")
    print(f"Average accuracy: {np.mean(accuracy_list):.1f}%")
    print(f"Average naive accuracy: {np.mean(naive_accuracy_list):.1f}%")
    print(f"Improvement over naive: {np.mean(accuracy_list) - np.mean(naive_accuracy_list):.1f} percentage points")


if __name__ == "__main__":
    main()