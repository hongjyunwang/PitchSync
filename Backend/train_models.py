import re
import pickle
import datetime as dt
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import os
from pybaseball import statcast

class PitchModelTrainer:
    def __init__(self):
        self.fastball_pitches = ['FA', 'FF', 'FT', 'FC', 'FS', 'SI', 'SF']
        self.id_columns = ['game_pk', 'pitcher', 'batter']
        self.situation_features = ['stand', 'p_throws', 'inning', 'balls', 'strikes', 
                                 'on_1b', 'on_2b', 'on_3b', 'outs_when_up', 'pitch_number', 
                                 'fld_score', 'bat_score']
        self.prev_pitch_features = ['type', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z', 
                                  'release_speed', 'release_spin_rate']
        
    def load_data(self, start_date='2024-03-20', end_date='2024-11-02'):
        """Load raw Statcast data"""
        print(f"Loading data from {start_date} to {end_date}")
        outcome = ['pitch_type']
        columns = outcome + self.id_columns + self.situation_features + self.prev_pitch_features
        
        raw_data = statcast(start_dt=start_date, end_dt=end_date, verbose=0)
        print(f"Raw data shape: {raw_data.shape}")
        
        data = raw_data[columns]
        return data
    
    def prep_train_data(self, data):
        """Clean and prepare training data"""
        # Drop rows with no pitch type
        data = data[pd.notnull(data['pitch_type'])]
        print(f"After dropping null pitch types: {data.shape}")
        
        # Map pitch types to fastball (1) or off-speed (0)
        data['pitch_type'] = data['pitch_type'].apply(self._map_fastballs)
        
        # Convert ID columns to int
        for col in self.id_columns:
            data[col] = data[col].astype('int64')
        
        # Convert count columns to int
        for col in ['inning', 'balls', 'strikes', 'outs_when_up', 'pitch_number']:
            data[col] = data[col].astype('int64')
        
        # Cap extra innings at 9
        data['inning'] = data['inning'].apply(lambda x: 9 if x > 9 else x)
        
        # Create game_pitcher_id for grouping
        data['game_pitcher_id'] = data['game_pk'].astype(str) + '_' + data['pitcher'].astype(str)
        
        # Convert base runners to boolean
        data['on_1b'] = data['on_1b'].apply(lambda x: not np.isnan(x))
        data['on_2b'] = data['on_2b'].apply(lambda x: not np.isnan(x))
        data['on_3b'] = data['on_3b'].apply(lambda x: not np.isnan(x))
        
        # Create handedness feature
        data['pitch_bat_same_side'] = data['p_throws'] == data['stand']
        data.drop(['p_throws', 'stand'], axis=1, inplace=True)
        
        # Score differential
        data['score_diff'] = data['fld_score'] - data['bat_score']
        data.drop(['fld_score', 'bat_score'], axis=1, inplace=True)
        
        # Add previous pitch features
        for col in self.prev_pitch_features + ['pitch_type']:
            data[f'prev_{col}'] = data.groupby('game_pitcher_id')[col].shift(1)
        data.drop(self.prev_pitch_features, axis=1, inplace=True)
        
        # Fill missing previous pitch data
        data['prev_pitch_type'].fillna('UN', inplace=True)
        data['prev_type'].fillna('UN', inplace=True)
        
        # Fill missing numerical features with pitcher means
        for col in self.prev_pitch_features:
            if col != 'type':
                prev_col = f"prev_{col}"
                if prev_col in data.columns:
                    data[prev_col] = data[prev_col].astype('float64')
                    data[prev_col] = data.groupby("game_pitcher_id")[prev_col].transform(
                        lambda x: x.fillna(x.mean())
                    )
        
        # Calculate fastball probabilities by game state
        cols_to_groupby = ['pitcher', 'outs_when_up', 'balls', 'strikes', 'on_1b', 'on_2b', 'on_3b']
        data_fb_prob = data.groupby(cols_to_groupby)['pitch_type'].mean().reset_index()
        data_fb_prob.rename(columns={'pitch_type': 'fb_prob'}, inplace=True)
        data = pd.merge(data, data_fb_prob, how='left', on=cols_to_groupby)
        
        # Drop remaining NAs
        data.dropna(inplace=True)
        
        # One-hot encode categorical features
        cols_to_encode = ['inning', 'balls', 'strikes', 'outs_when_up', 'prev_type', 'prev_pitch_type']
        data = pd.get_dummies(data, prefix=cols_to_encode, columns=cols_to_encode)
        
        # Drop ID columns not needed for training
        data.drop(['game_pk', 'batter', 'game_pitcher_id'], axis=1, inplace=True)
        
        return data
    
    def train_models(self, data, pitch_count_cutoff=1000):
        """Train individual models for each pitcher"""
        # Ensure directory exists
        os.makedirs('./models/pitcher_models/', exist_ok=True)
        
        # Get pitcher counts
        pitcher_count_dict = dict(Counter(data['pitcher']))
        pitcher_count_dict = {k: v for k, v in pitcher_count_dict.items() if v > pitch_count_cutoff}
        
        print(f"Training models for {len(pitcher_count_dict)} pitchers")
        
        accuracy_list = []
        naive_accuracy_list = []
        
        for i, pitcher in enumerate(pitcher_count_dict.keys()):
            start = dt.datetime.now()
            
            # Get pitcher data
            df_pitcher = data[data['pitcher'] == pitcher].copy()
            df_pitcher.drop('pitcher', axis=1, inplace=True)
            
            # Split features and target
            X = df_pitcher.drop('pitch_type', axis=1)
            y = df_pitcher['pitch_type']
            
            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=4256
            )
            
            # XGBoost with grid search
            xgb_params = {
                "max_depth": [2, 5, 20],
                "learning_rate": [0.01, 0.1, 0.4]
            }
            
            xgb_opt = GridSearchCV(
                XGBClassifier(random_state=42), 
                param_grid=xgb_params, 
                cv=5, 
                scoring='accuracy', 
                verbose=0, 
                n_jobs=-1
            )
            
            xgb_opt.fit(X_train, y_train)
            y_pred = xgb_opt.predict(X_test)
            
            accuracy = round(accuracy_score(y_test, y_pred) * 100, 1)
            naive_accuracy = df_pitcher['fb_prob'].mean() * 100
            
            accuracy_list.append(accuracy)
            naive_accuracy_list.append(naive_accuracy)
            
            # Save model with metadata
            pitcher_model = {
                'model': xgb_opt.best_estimator_,
                'feature_names': X.columns.tolist(),
                'model_accuracy': accuracy,
                'naive_accuracy': naive_accuracy,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'best_params': xgb_opt.best_params_
            }
            
            model_path = f'./models/pitcher_models/{pitcher}.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(pitcher_model, f)
            
            if i % 10 == 0:
                print(f"Pitcher {pitcher}: {accuracy}% accuracy (vs {naive_accuracy:.1f}% naive)")
                print(f"Training time: {dt.datetime.now() - start}")
        
        return accuracy_list, naive_accuracy_list
    
    def _map_fastballs(self, pitch_type):
        """Map pitch types to fastball (1) or off-speed (0)"""
        return 1 if pitch_type in self.fastball_pitches else 0

def main():
    """Main training script"""
    trainer = PitchModelTrainer()
    
    # Load and prepare data
    raw_data = trainer.load_data()
    clean_data = trainer.prep_train_data(raw_data)
    
    # Train models
    accuracy_list, naive_accuracy_list = trainer.train_models(clean_data)
    
    # Print summary
    improvement = [acc - naive for acc, naive in zip(accuracy_list, naive_accuracy_list)]
    positive_improvements = [x for x in improvement if x > 0]
    improve_pct = len(positive_improvements) / len(improvement) * 100
    
    print(f"\nTraining Summary:")
    print(f"Models trained: {len(accuracy_list)}")
    print(f"Average accuracy: {np.mean(accuracy_list):.1f}%")
    print(f"Average naive accuracy: {np.mean(naive_accuracy_list):.1f}%")
    print(f"Models with improvement: {improve_pct:.1f}%")

if __name__ == "__main__":
    main()


