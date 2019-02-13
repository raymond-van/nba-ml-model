import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from data import X, y, mean_team_stats
from train import rf
# from calibration import rf_isotonic, xgb_sigmoid
# from ensemble import xgb_stack
from schedule import games, num_of_games
from pymongo import MongoClient
import json
from pprint import pprint

today_df = pd.DataFrame()

# Turn games into a dataframe that is in the same format as training data
# The model makes predictions by comparing the average stats of team_a and team_b
# Returns the win probability for team_a (road_team)
for game in games:
    road_team, home_team = game[0],game[1]
    team_a = mean_team_stats.loc[mean_team_stats['TEAM_NAME'] == road_team]
    team_a.drop(['TEAM_ID','TEAM_NAME'],axis=1,inplace=True)
    team_b = mean_team_stats.loc[mean_team_stats['TEAM_NAME'] == home_team]
    team_b.drop(['TEAM_ID','TEAM_NAME'],axis=1,inplace=True)
    cols_a = [col + 'A' for col in team_a.columns]
    cols_b = [col + 'B' for col in team_b.columns]
    team_a.columns = cols_a
    team_a.reset_index(drop=True,inplace=True)
    team_b.columns = cols_b
    team_b.reset_index(drop=True,inplace=True)
    game = pd.concat([team_a,team_b],axis=1)
    today_df = today_df.append(game)
    today_df.reset_index(drop=True,inplace=True)

# Add home court adv. columns
today_df['TEAMA_HCA'] = pd.Series([0]*num_of_games, index=today_df.index)
today_df['TEAMB_HCA'] = pd.Series([1]*num_of_games, index=today_df.index)
today_df = today_df.reindex_axis(sorted(today_df.columns), axis=1)   

# Model prediction
rf_prob = rf.predict_proba(today_df)
rf_prob = rf_prob[:,1]

# rf_iso_prob = rf_isotonic.predict_proba(today_df)
# rf_prob = rf_iso_prob[:,1]
# xgb_sigmoid_prob = xgb_sigmoid.predict_proba(today_df)
# rf_prob = xgb_sigmoid_prob[:,1]
# xgb_stack_prob = xgb_stack.predict_proba(today_df)
# print(xgb_stack_prob[:,1])

# Process predictions for neat display in dataframe
rf_pred = []
for prob in rf_prob:
    if prob < .5:
        rf_pred.append(0)
    else:
        rf_pred.append(1)
games_data = np.empty((len(games),3))
games_data[:,0] = rf_pred
games_data[:,1] = np.around(rf_prob*100,2)
games_data[:,2] = np.around(norm(0,10.5).ppf(rf_prob),1)    # Convert win probability into point spread
games_str = []
for game in games:      # turn games list of lists into list of strings for dataframe index
    t1 = game[0]
    t2 = game[1]
    teams = t1 + ' at ' + t2
    games_str.append(teams)

games_df = pd.DataFrame(data=games_data,index=games_str,columns=['Road Team Win','Road Team Win Prob','Road Team Point Spread'])
print(games_df)

###########################################################

# Send predictions output to a database

# Convert pandas dataframe to JSON
games_json = games_df.to_dict(orient='index')
# pprint(games_json)
# pprint(type(games_json))

# Connect to MongoDB and create database
client = MongoClient("")
db = client.nba
result = db.predictions.insert_one(games_json)