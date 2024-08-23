from IPython.display import Image
import re
import ast
import numpy as np
import pandas as pd
import json
import pickle
from collections import Counter
import datetime as dt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import requests

def prep_train_data(df):
    '''
    Function to select and clean features and generate some new features
    
    df: raw data on pitching 
    returns a cleaned dataframe ready for training
    '''
    
    # select the columns to keep
    cols_to_keep = ['game_pk', 'pitcher_id', 'pitch_type', 'inning', 'top', 'pcount_at_bat', 'pcount_pitcher', 
                    'balls', 'strikes', 'fouls', 'outs', 'on_1b', 'on_2b', 'on_3b', 
                    'b_height', 'away_team_runs', 'home_team_runs', 
                    'p_throws', 'stand', 'type', 'vy0', 'break_length', 'break_angle',  'zone', 'date']
    df = df[cols_to_keep]
    
    # drop any missing values in the outcome column ("pitch_type") 
    df = df[~pd.isna(df['pitch_type'])]

    # extract the month from the date for seasonality and drop date column
    df['month'] = df['date'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d').month)
    df.drop('date', axis=1, inplace=True)
        
    # convert on_1b/on_2b/on_3b to boolean 
    df['on_1b'] = df['on_1b'].apply(lambda x: not np.isnan(x))
    df['on_2b'] = df['on_2b'].apply(lambda x: not np.isnan(x))
    df['on_3b'] = df['on_3b'].apply(lambda x: not np.isnan(x))

    # handedness: does the batter hit from the same side that the pitcher is pitching from
    df['stand_pitch_same_side'] = df['p_throws'] == df['stand'] 
    df.drop(['p_throws', 'stand'], axis=1, inplace=True)
    
    # score differential
    df['score_diff'] = -np.power(-1, df['top']) * (df['home_team_runs'] - df['away_team_runs'])
    df.drop(['home_team_runs', 'away_team_runs'], axis=1, inplace=True)
    
    # make a new id based on game id + pitcher id that we can use for groupby's
    df['game_pitcher_id'] = df['game_pk'].astype(str) + '_' + df['pitcher_id'].astype(str)
    df.drop('game_pk', axis=1, inplace=True)

    # get previous pitch type to use as a feature
    df['prev_pitch_type'] = df.groupby('game_pitcher_id')['pitch_type'].apply(lambda x: x.shift(1))

    # get previous pitch outcome to use as a feature
    df['prev_pitch_outcome'] = df.groupby('game_pitcher_id')['type'].apply(lambda x: x.shift(1))
    df.drop('type', axis=1, inplace=True)

    # get previous pitch z position to use as a feature
    df['prev_zone'] = df.groupby('game_pitcher_id')['zone'].apply(lambda x: x.shift(1))
    df.drop('zone', axis=1, inplace=True)

    # get previous break length to use as a feature
    df['prev_break_length'] = df.groupby('game_pitcher_id')['break_length'].apply(lambda x: x.shift(1))
    df.drop('break_length', axis=1, inplace=True)

    # get previous break angle to use as a feature
    df['prev_break_angle'] = df.groupby('game_pitcher_id')['break_angle'].apply(lambda x: x.shift(1))
    df.drop('break_angle', axis=1, inplace=True)

    # get previous pitch velocity to use as a feature (and convert to mph)
    df['prev_pitch_velocity'] = df.groupby('game_pitcher_id')['vy0'].apply(lambda x: x.shift(1))
    df['prev_pitch_velocity'] = round(-df['prev_pitch_velocity'] * (3600 / 5280), 0)
    df.drop('vy0', axis=1, inplace=True)

    # fill the missing prev_pitch_type and  with an Unknown token
    df['prev_pitch_type'].fillna('UN', inplace=True)
    df['prev_pitch_outcome'].fillna('UN', inplace=True)
    
    # fill missing prev_pitch_velocity with pitcher's mean velocity
    df["prev_pitch_velocity"] = df.groupby("game_pitcher_id")["prev_pitch_velocity"].transform(lambda x: x.fillna(x.mean()))
    
    # convert height of batter to inches
    def convert_height(x):
        feet = re.findall('^[0-9]', x)[0]
        inches = re.findall('.[0-9]$', x)[0]
        return 12 * int(feet) + int(inches)
    df['batter_height'] = df['b_height'].apply(convert_height)
    df.drop('b_height', axis=1, inplace=True)
    
    # drop any rows with pitchouts or unknown pitches
    pitchout_unknown_pitches = ['PO', 'FO', 'UN', 'XX', 'IN']
    df = df[~df['pitch_type'].isin(pitchout_unknown_pitches)]
    df = df[~df['prev_pitch_type'].isin(pitchout_unknown_pitches)]
    
    # map all fastball pitches into one pitch type (FB)
    fastball_pitches = ['FA', 'FF', 'FT', 'FC', 'FS', 'SI', 'SF']
    def map_fastballs(x):
        if x in fastball_pitches:
            return 'FB'
        else:
            return x
    df['pitch_type'] = df['pitch_type'].apply(map_fastballs)
    df['prev_pitch_type'] = df['prev_pitch_type'].apply(map_fastballs)

    # map previous pitch outcomes to 0 (ball, B), 1 (strike, S), 2 (in-play, X)
    def map_pitch_outcome(x):
        if x == 'B':
            return 0
        elif x == 'S':
            return 1
        else:
            return 2
    df['prev_pitch_outcome'] = df['prev_pitch_outcome'].apply(map_pitch_outcome)
    
    # reorganize the column order
    df = df[['pitcher_id', 'pitch_type', 'inning', 'top', 'pcount_at_bat', 'pcount_pitcher', 
             'balls', 'strikes', 'fouls', 'outs', 'on_1b', 'on_2b', 'on_3b', 'month', 
             'stand_pitch_same_side', 'score_diff', 'prev_pitch_type', 
             'prev_pitch_outcome', 'prev_zone', 'prev_break_length', 'prev_break_angle', 
             'prev_pitch_velocity', 'batter_height']]
    
    return df


