import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
pd.set_option('display.max_columns', None)

# Load and process game data used for training
X = pd.read_feather(os.getcwd() + '/data/nba_data.feather')
y = X['TEAMA_WIN']
X = X.drop(['GAME_IDA', 'TEAM_NAMEA','TEAM_ABBREVIATIONA','TEAM_CITYA','MINA',
              'FGMA','FGAA','FG3MA','FG3AA','PTSA','USG_PCTA','E_USG_PCTA',
              'GAME_IDB', 'TEAM_NAMEB','TEAM_ABBREVIATIONB','TEAM_CITYB','MINB',
              'FGMB','FGAB','FG3MB','FG3AB','PTSB','USG_PCTB','E_USG_PCTB', 'HOME_TEAM_ID',
              'TEAM_IDA','TEAM_IDB','E_OFF_RATINGA','E_OFF_RATINGB','FTMA','FTMB','FTAA','FTAB',
              'E_DEF_RATINGA','E_DEF_RATINGB','E_NET_RATINGA','E_NET_RATINGB','E_TM_TOV_PCTA','E_TM_TOV_PCTB',
              'E_PACEA','E_PACEB','PLUS_MINUSA','PLUS_MINUSB', 'NET_RATINGA','NET_RATINGB','TEAMA_WIN','PIEA','PIEB'], axis=1)
# Dropping unimportant columns              
X = X.drop(['AST_PCTA','AST_PCTB','BLKA','BLKB','FTA_RATEA','FTA_RATEB','FT_PCTA','FT_PCTB','OPP_FTA_RATEA','OPP_FTA_RATEB','PACEA','PACEB','PFA','PFB','STLA','STLB','OREBA','OREBB'],axis=1) 
# Standardization
cols = X.columns
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = pd.DataFrame(data=X, columns=cols)
# Sort columns
X = X.reindex_axis(sorted(X.columns), axis=1) 

# Load and process average team stats used for predictions
team_raw = pd.read_feather(os.getcwd() + '/data/team_raw.feather')
mean_team_stats = team_raw.groupby(['TEAM_NAME'],as_index=False).mean()
mean_team_stats = mean_team_stats.drop(['FGM','FGA','FG3M','FG3A','FTM','FTA','PTS','PLUS_MINUS','E_OFF_RATING','E_DEF_RATING',
                      'E_NET_RATING','NET_RATING','E_TM_TOV_PCT','USG_PCT','E_USG_PCT','E_PACE','PIE'],axis=1)
# Dropping unimportant columns                      
mean_team_stats = mean_team_stats.drop(['AST_PCT','BLK','FTA_RATE','FT_PCT','OPP_FTA_RATE','PACE','PF','STL','OREB'],axis=1)
# Remove columns w/ strings for standardization
team_ids = mean_team_stats.iloc[:,[0,1]]
mean_team_stats = mean_team_stats.drop(['TEAM_NAME','TEAM_ID'],axis=1)
# Standardization
cols = mean_team_stats.columns
scaler = StandardScaler()
mean_team_stats = scaler.fit_transform(mean_team_stats)
mean_team_stats = pd.DataFrame(mean_team_stats, columns=cols)
# Re-add team ids
mean_team_stats['TEAM_NAME'] = team_ids['TEAM_NAME']
mean_team_stats['TEAM_ID'] = team_ids['TEAM_ID']