import pandas as pd
import numpy as np
from scipy.stats import norm
from data_unprocessed import X, y, mean_team_stats
from train import rf, et
from schedule import games, num_of_games
from pymongo import MongoClient

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
et_prob = et.predict_proba(today_df)
et_prob = et_prob[:,1]
avg_prob = (rf_prob + et_prob) / 2

# Process predictions for neat display in dataframe
avg_pred = []
for prob in avg_prob:
    if prob < .5:
        avg_pred.append(0)
    else:
        avg_pred.append(1)
games_data = np.empty((len(games),3))
games_data[:,0] = avg_pred
games_data[:,1] = np.around(avg_prob*100,2)
games_data[:,2] = np.around(norm(0,10.5).ppf(avg_prob),1)    # Convert win probability into point spread
games_str = []
for game in games:      # turn games list of lists into list of strings for dataframe indexing
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

# Connect to MongoDB and create database
# client = MongoClient("")
# db = client.<db>
# result = db.<collection>.insert_one(games_json)