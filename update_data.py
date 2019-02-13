#%%
from nba_py import game
from nba_py import constants
from nba_py import team
import nba_py
import requests
import requests_cache
import numpy as np
import pandas as pd
import os
pd.set_option('display.max_columns', None)
# LATEST GAME IN DATA.. ID=0021800847
#%%
# Get ID of most recent game
check_id = '0021800735'
foo = game.Boxscore(check_id)
foo = foo.team_stats()
# print(foo)

#%%
# Collect and process data of the latest games played currently not used in the model
g_id = '0021800736'
trad = game.Boxscore(g_id)
trad = trad.team_stats()
adv = game.BoxscoreAdvanced(g_id)
adv = adv.sql_team_advanced()
ff = game.BoxscoreFourFactors(g_id)
ff = ff.sql_team_four_factors()
df = []
df.append(trad); df.append(adv); df.append(ff)
merge0 = pd.concat(df,axis=1)
df1, df2 = np.split(merge0,[1],axis=0)
cols_A = [col + 'A' for col in df1.columns]
cols_B = [col + 'B' for col in df2.columns]
df1.columns = cols_A
df2.columns = cols_B
df2 = df2.reset_index(drop=True)
summ = game.BoxscoreSummary(g_id)
hc_df = summ.game_summary()
hc_df = hc_df[['HOME_TEAM_ID']]
raw_data = pd.concat([df1,df2,hc_df],axis=1)
for i in range(21800737,21800848,1):
    game_id = '00' + str(i)
    trad = game.Boxscore(game_id)
    trad = trad.team_stats()
    adv = game.BoxscoreAdvanced(game_id)
    adv = adv.sql_team_advanced()
    ff = game.BoxscoreFourFactors(game_id)
    ff = ff.sql_team_four_factors()
    df = []
    df.append(trad); df.append(adv); df.append(ff)
    merge = pd.concat(df,axis=1)
    merge0 = pd.concat([merge0,merge],axis=0)
    df1, df2 = np.split(merge,[1],axis=0)
    cols_A = [col + 'A' for col in df1.columns]
    cols_B = [col + 'B' for col in df2.columns]
    df1.columns = cols_A
    df2.columns = cols_B
    df2 = df2.reset_index(drop=True)
    df = pd.concat([df1,df2],axis=1)
    summ = game.BoxscoreSummary(game_id)
    summ = summ.game_summary()
    summ = summ[['HOME_TEAM_ID']]
    df = pd.concat([df,summ],axis=1)
    raw_data = pd.concat([raw_data,df],axis=0)
    raw_data.reset_index(drop=True,inplace=True)

#%%
print(raw_data)

#%%
# Process raw game data
raw_data = raw_data.iloc[:,~raw_data.columns.duplicated()]

#%%
# Add home court adv. and w/l as columns in new data
def win_lose(x):
    if x < 0:
        return 0
    else:
        return 1
def court_advA(x,a):
    if x == a:
        return 1
    else:
        return 0
def court_advB(x,b):
    if x == b:
        return 1
    else:
        return 0
raw_data['TEAMA_WIN'] = raw_data.apply(lambda row: win_lose(row['PLUS_MINUSA']), axis=1)
raw_data['TEAMA_HCA'] = raw_data.apply(lambda row: court_advA(row['HOME_TEAM_ID'],row['TEAM_IDA']), axis=1)
raw_data['TEAMB_HCA'] = raw_data.apply(lambda row: court_advB(row['HOME_TEAM_ID'],row['TEAM_IDB']), axis=1)

#%%
# Load old game data
data = pd.read_feather(os.getcwd() + '/data/nba_data.feather')

#%%
# Merge old game data with new game data
data = pd.concat([data,raw_data],axis=0)
data.reset_index(drop=True,inplace=True)

#%%
data.to_feather(os.getcwd() + '/data/nba_data.feather')

#%%
# Process raw mean team stats
merge0 = merge0.iloc[:,~merge0.columns.duplicated()]
merge0.reset_index(drop=True,inplace=True)

#%%
# Load old team stats
team_raw = pd.read_feather(os.getcwd() + '/data/team_raw.feather')

#%%
# Merge old and new team stats
team_raw = pd.concat([team_raw,merge0],axis=0)
team_raw.reset_index(drop=True,inplace=True)

#%%
team_raw.to_feather(os.getcwd() + '/data/team_raw.feather')