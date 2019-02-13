import pandas as pd
import os
pd.set_option('display.max_columns', None)

# Load game data w/ some processing
X = pd.read_feather(os.getcwd() + '/data/nba_data.feather')
y = X['TEAMA_WIN']
X = X.drop(['GAME_IDA', 'TEAM_NAMEA','TEAM_ABBREVIATIONA','TEAM_CITYA','MINA',
              'FGMA','FGAA','FG3MA','FG3AA','PTSA','USG_PCTA','E_USG_PCTA',
              'GAME_IDB', 'TEAM_NAMEB','TEAM_ABBREVIATIONB','TEAM_CITYB','MINB',
              'FGMB','FGAB','FG3MB','FG3AB','PTSB','USG_PCTB','E_USG_PCTB', 'HOME_TEAM_ID',
              'TEAM_IDA','TEAM_IDB','E_OFF_RATINGA','E_OFF_RATINGB','FTMA','FTMB','FTAA','FTAB',
              'E_DEF_RATINGA','E_DEF_RATINGB','E_NET_RATINGA','E_NET_RATINGB','E_TM_TOV_PCTA','E_TM_TOV_PCTB',
              'E_PACEA','E_PACEB','PLUS_MINUSA','PLUS_MINUSB', 'NET_RATINGA','NET_RATINGB','TEAMA_WIN','PIEA','PIEB'], axis=1)
X = X.reindex_axis(sorted(X.columns), axis=1)  

# Average team stats used for predictions
team_raw = pd.read_feather(os.getcwd() + '/data/team_raw.feather')
mean_team_stats = team_raw.groupby(['TEAM_NAME'],as_index=False).mean()
mean_team_stats.drop(['FGM','FGA','FG3M','FG3A','FTM','FTA','PTS','PLUS_MINUS','E_OFF_RATING','E_DEF_RATING',
                      'E_NET_RATING','NET_RATING','E_TM_TOV_PCT','USG_PCT','E_USG_PCT','E_PACE','PIE'],axis=1,inplace=True)
