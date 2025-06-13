import re
import pickle
from IPython.display import Image

from pybaseball import statcast, pitching_stats
import datetime as dt
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from matplotlib import patches

import os
import json


# use Statcast data (from 2015-2018) so we can get spin rate, etc.
train_data_dates = [('2024-03-20', '2024-11-02')]#,      # 2015 data
#                     ('2016-04-03', '2016-10-02'),       # 2016 data
#                     ('2017-04-02', '2017-10-01'),       # 2017 data
#                     ('2018-03-29', '2018-10-01')]       # 2018 data
# ------------------------------------ build the dataframe ------------------------------------

raw_data = statcast(start_dt=train_data_dates[0][0], end_dt=train_data_dates[0][1], verbose=0)
print(raw_data.shape)
raw_data.head()

# ------------------------------------ select the outcome and features ------------------------------------

outcome = ['pitch_type']

id_columns = ['game_pk', 'pitcher', 'batter']

situation_features = ['stand', 'p_throws', 'inning', 'balls', 'strikes', 
                      'on_1b', 'on_2b', 'on_3b', 'outs_when_up', 'pitch_number', 
                      'fld_score', 'bat_score']

prev_pitch_features = ['type', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z', 'release_speed', 'release_spin_rate']

data = raw_data[outcome + id_columns + situation_features + prev_pitch_features]

print(data.shape)
data.head()

# ------------------------------------ clean up and feature engineering ------------------------------------

def prep_train_data(data):
    '''
    Function to select and clean features and generate some new features
    
    df: raw data on pitching 
    cleans dataframe to prepare for training
    '''

    # drop all columns with no pitch type categorization
    data = data[pd.notnull(data['pitch_type'])]
    print(data.shape)
    data.head()

    # categorize pitch types as "Fastball" (1) or "Off-speed" (0)
    fastball_pitches = ['FA', 'FF', 'FT', 'FC', 'FS', 'SI', 'SF']
    def map_fastballs(x):
        if x in fastball_pitches:
            return 1
        else:
            return 0
    data['pitch_type'] = data['pitch_type'].apply(map_fastballs)

    data.head()

    # make sure ID columns are int's - use regular int, not nullable Int64
    for col in id_columns:
        data[col] = data[col].astype('int64')  # Use regular int64, not nullable Int64
    # convert innings, balls and strikes to ints - use regular int, not nullable Int64
    for col in ['inning', 'balls', 'strikes', 'outs_when_up', 'pitch_number']:
        data[col] = data[col].astype('int64')  # Use regular int64, not nullable Int64

    # if inning > 9, just replace with "9"
    def cap_extra_innings(x):
        if x > 9:
            return 9
        else:
            return x
    data['inning'] = data['inning'].apply(cap_extra_innings)
        
    # make a new id based on game id + pitcher id that we can use for groupby's
    data['game_pitcher_id'] = data['game_pk'].astype(str) + '_' + data['pitcher'].astype(str)

    # convert on_1b/on_2b/on_3b to boolean 
    data['on_1b'] = data['on_1b'].apply(lambda x: not np.isnan(x))
    data['on_2b'] = data['on_2b'].apply(lambda x: not np.isnan(x))
    data['on_3b'] = data['on_3b'].apply(lambda x: not np.isnan(x))

    # handedness: does the batter hit from the same side that the pitcher is pitching from
    data['pitch_bat_same_side'] = data['p_throws'] == data['stand'] 
    data.drop(['p_throws', 'stand'], axis=1, inplace=True)

    # score differential
    data['score_diff'] = data['fld_score'] - data['bat_score']
    data.drop(['fld_score', 'bat_score'], axis=1, inplace=True)

    data.head()

    # groupby on game_pitcher_id and tabulate previous pitch data
    for col in prev_pitch_features + outcome:
        data[f'prev_{col}'] = data.groupby('game_pitcher_id')[col].shift(1)
    data.drop(prev_pitch_features, axis=1, inplace=True)

    # fill the missing prev_pitch_type with an Unknown token
    data['prev_pitch_type'].fillna('UN', inplace=True)
    data['prev_type'].fillna('UN', inplace=True)

    print(data.shape)
    data.head()

    # fill missing prev_pitch_velocity with pitcher's mean velocity
    for col in prev_pitch_features:
        if col != 'type':
            prev_col = f"prev_{col}"
            if prev_col in data.columns:
                print(f"Filling missing values for {prev_col}")
                # Convert to float first to handle mixed int/float filling
                data[prev_col] = data[prev_col].astype('float64')
                data[prev_col] = data.groupby("game_pitcher_id")[prev_col].transform(lambda x: x.fillna(x.mean()))

    print(data.shape)
    data.head()

    for col in data.columns.tolist():
        print(col, data[col].tolist()[:10])

prep_train_data(data)

# ------------------------------------ pitch probabilities for given state of at-bat ------------------------------------

cols_to_groupby = ['pitcher', 'outs_when_up', 'balls', 'strikes', 'on_1b', 'on_2b', 'on_3b']

data_fb_prob = pd.DataFrame(data.groupby(cols_to_groupby)['pitch_type'].mean())

data_fb_prob.rename(columns={'pitch_type': 'fb_prob'}, inplace=True)

data_fb_prob.reset_index(inplace=True, drop=False)

data = pd.merge(data, data_fb_prob, how='left', on=cols_to_groupby)

print(data.shape)
data.head()

# drop any remaining NA's
data.dropna(inplace=True)

for col in data.columns.tolist():
    print(col, data[col].tolist()[:5])

cols_to_encode = ['inning', 'balls', 'strikes', 'outs_when_up', 'prev_type', 'prev_pitch_type']

data = pd.get_dummies(data, prefix=cols_to_encode, columns=cols_to_encode)

data.head()

data.drop(['game_pk', 'batter', 'game_pitcher_id'], axis=1, inplace=True)



def train_models(train_data, pitch_count_cutoff=1000):
    '''
    Function to train and test models for pitch prediction for individual pitchers
    
    train_data: a cleaned data frame
    pitch_count_cutoff: a minimum number of pitches thrown  
    
    returns a pickled file for each pitcher that contains the model and some metadata for that pitcher
    '''

    # Ensure directory exists
    os.makedirs('./data/pitcher_models/', exist_ok=True)

    # build a dict with pitch_id as key and total pitch count as value
    pitcher_count_dict = dict(Counter(train_data['pitcher']))

    # drop pitchers that don't have enough pitches to build a reliable model
    pitcher_count_dict = {k:v for k, v in pitcher_count_dict.items() if v > pitch_count_cutoff}

    # list of pitchers
    pitcher_list = pitcher_count_dict.keys()
    print(f"Number of pitchers that make the cut: {len(pitcher_count_dict)}")

    # loop through the list of pitchers and train models
    accuracy_list = []
    naive_accuracy_list = []
    num_skipped = 0
    for i, pitcher in enumerate(pitcher_list):

        # start a timer
        start = dt.datetime.now()

        # subsets the dataframe to only include the current pitcher's data
        df_pitcher = train_data[train_data['pitcher'] == pitcher]
        df_pitcher.drop('pitcher', axis=1, inplace=True)

        # Extract and map pitch types to ints
        unique_pitches = sorted(df_pitcher['pitch_type'].unique())
        pitch_map = {pitch: idx for idx, pitch in enumerate(unique_pitches)}
        pitch_unmap = {idx: pitch for pitch, idx in pitch_map.items()}
        df_pitcher['pitch_type'] = df_pitcher['pitch_type'].map(pitch_map)

        # split the dataframe into a feature set and an outcome column
        X = df_pitcher.drop('pitch_type', axis=1)
        y = df_pitcher['pitch_type']

        # split the data into train/test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4256)

        # ----------------------
        # train an XGBoost model
        # ----------------------

        # small set of hyperparameters to optimize over
        xgb_params = {"max_depth": (2, 5, 20),
                      "learning_rate": (0.01, 0.1, 0.4)}

        # perform the paramater grid search using 5-fold cross validation
        xgb_opt = GridSearchCV(XGBClassifier(), 
                               param_grid=xgb_params, 
                               cv=5, 
                               scoring='accuracy', 
                               verbose=0, 
                               n_jobs=-1)

        # perform fit and make predictions
        xgb_opt.fit(X_train, y_train)
        y_pred = xgb_opt.predict(X_test)
        y_prob = xgb_opt.predict_proba(X_test)

        # compute accuracy and store in a list for analyzing results later
        accuracy = round(accuracy_score(y_test, y_pred) * 100, 1)
        accuracy_list.append(accuracy)

        # get and store the naive accuracy (accuracy from just predicting the most thrown pitch)
        naive_accuracy = df_pitcher['fb_prob'].mean()
        naive_accuracy_list.append(naive_accuracy)

        # Save model with metadata
        pitcher_model = {
            'model': xgb_opt.best_estimator_,
            'pitch_map': pitch_map,
            'pitch_unmap': pitch_unmap,
            'model_accuracy': accuracy
        }

        model_path = f'./data/pitcher_models/{pitcher}.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(pitcher_model, f)

        # print some input/results for every 10th pitcher
        if i % 10 == 0:
            print()
            print(f"Pitcher ID: {pitcher}")
            print(f"Number of data points in training: {X_train.shape[0]}")
            print(f"Number of data points in testing: {X_test.shape[0]}")
            print(f"Best params: {xgb_opt.best_params_}")
            print(f"Total training time: {dt.datetime.now()-start}")
            print(f"Naive accuracy: {naive_accuracy}")
            print(f"XGBooost accuracy: {accuracy}")

    # return the accuracy lists so we can perform assessment 
    return accuracy_list, naive_accuracy_list



def perform_prediction(input_data):
    """
    Function used to perform real-time predictions using the trained model
    The function should take an input data frame that describes a precise game scenario, clean 
    the dataframe to keep only relevant parameters, and predict whether the next pitch will be 
    a fastball or not. 
    """
    # Write the code
    if input_data.shape[0] != 1:
        raise ValueError("Input must be a single-row DataFrame representing one pitch scenario.")

    # Extract pitcher ID
    pitcher_id = input_data.iloc[0]['pitcher']
    model_path = f'./data/pitcher_models/{pitcher_id}.pkl'
    
    # Load model
    if not os.path.exists(model_path):
        raise ValueError(f"No trained model found for pitcher ID {pitcher_id}")

    with open(model_path, 'rb') as f:
        pitcher_model = pickle.load(f)

    model = pitcher_model['model']
    pitch_map = pitcher_model['pitch_map']
    pitch_unmap = pitcher_model['pitch_unmap']

    # Use the same preprocessing pipeline used during training
    processed = prep_single_pitch(input_data.copy())

    # Align columns with model training (some get_dummies columns may be missing)
    expected_features = model.get_booster().feature_names
    for col in expected_features:
        if col not in processed.columns:
            processed[col] = 0  # add missing columns with 0s
    processed = processed[expected_features]

    # Make prediction
    prediction = model.predict(processed)[0]
    probability = model.predict_proba(processed)[0][prediction]

    result = {
        "prediction": pitch_unmap[prediction],
        "is_fastball": True if pitch_unmap[prediction] == 1 else False,
        "probability": probability
    }

    return result

def prep_single_pitch(data):
    # Use same logic from prep_train_data
    fastball_pitches = ['FA', 'FF', 'FT', 'FC', 'FS', 'SI', 'SF']
    def map_fastballs(x):
        if x in fastball_pitches:
            return 1
        else:
            return 0

    if 'pitch_type' in data.columns:
        data['pitch_type'] = data['pitch_type'].apply(map_fastballs)

    # Basic type casting
    for col in ['game_pk', 'pitcher', 'batter']:
        data[col] = data[col].astype('int64')
    for col in ['inning', 'balls', 'strikes', 'outs_when_up', 'pitch_number']:
        data[col] = data[col].astype('int64')

    data['inning'] = data['inning'].apply(lambda x: 9 if x > 9 else x)
    data['game_pitcher_id'] = data['game_pk'].astype(str) + '_' + data['pitcher'].astype(str)
    data['on_1b'] = data['on_1b'].apply(lambda x: not np.isnan(x))
    data['on_2b'] = data['on_2b'].apply(lambda x: not np.isnan(x))
    data['on_3b'] = data['on_3b'].apply(lambda x: not np.isnan(x))

    data['pitch_bat_same_side'] = data['p_throws'] == data['stand']
    data.drop(['p_throws', 'stand'], axis=1, inplace=True)

    data['score_diff'] = data['fld_score'] - data['bat_score']
    data.drop(['fld_score', 'bat_score'], axis=1, inplace=True)

    # Fill or drop missing prev_ columns
    for col in ['type', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z', 'release_speed', 'release_spin_rate']:
        prev_col = f'prev_{col}'
        if prev_col not in data.columns:
            data[prev_col] = 0.0
        else:
            data[prev_col].fillna(0.0, inplace=True)

    for col in ['prev_pitch_type', 'prev_type']:
        if col not in data.columns:
            data[col] = 'UN'
        else:
            data[col].fillna('UN', inplace=True)

    # Add fb_prob
    data['fb_prob'] = 0.5  # Default; ideally use precomputed fb_prob lookup

    # One-hot encoding
    cols_to_encode = ['inning', 'balls', 'strikes', 'outs_when_up', 'prev_type', 'prev_pitch_type']
    data = pd.get_dummies(data, prefix=cols_to_encode, columns=cols_to_encode)

    # Drop IDs not needed
    data.drop(['game_pk', 'batter', 'game_pitcher_id'], axis=1, inplace=True)

    return data


accuracy_list, naive_accuracy_list = train_models(data, pitch_count_cutoff=1000)

accuracy_diff_list = [accuracy_list[i] - naive_accuracy_list[i] for i in range(len(accuracy_list))]
pos_diff_list = [x for x in accuracy_diff_list if x > 0.]
improve_pct = round(len(pos_diff_list) / len(accuracy_diff_list) * 100., 1)
print(f"Percentage of pitcher models with accuracy improvement over naive prediction: {improve_pct}%")

fig = plt.figure(figsize=(12,5))
plt.subplot(1, 2, 1)
plt.hist(accuracy_diff_list, bins=10)
plt.title("Accuracy Improvement with Model")
plt.ylabel("Frequency")
plt.xlabel("Accuracy")

plt.subplot(1, 2, 2)
plt.hist(naive_accuracy_list, label='naive')
plt.hist(accuracy_list, alpha=0.5, label='model')
plt.title("Distribution of Model Predictions vs. Naive Predictions")
plt.ylabel("Frequency")
plt.xlabel("Accuracy")
plt.legend(loc='best')
plt.show()






# To test

# Pull recent pitch data from Statcast (e.g., Gerrit Cole game on 2024-04-10)
test_input = statcast(start_dt='2024-04-10', end_dt='2024-04-10')
test_input = test_input[test_input['pitcher'] == 543037]  # replace with your pitcher's ID
test_row = test_input.iloc[[10]]  # simulate predicting after the 10th pitch

result = perform_prediction(test_row)
print(result)

