import pandas as pd
import numpy as np
from scipy.stats import norm
from data_unprocessed import X, y, mean_team_stats
from train import rf, xgb, et, sv, gb
from calibration import rf_isotonic, xgb_sigmoid, gb_sigmoid, sv_sigmoid, et_isotonic
from schedule import games, num_of_games
import warnings
warnings.filterwarnings("ignore")

today_df = pd.DataFrame()

# Similar code as predict.py but this is used to compare uncalibrated probabilities vs. calibrated probabilities
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
xgb_prob = xgb.predict_proba(today_df)
xgb_prob = xgb_prob[:,1]
gb_prob = gb.predict_proba(today_df)
gb_prob = gb_prob[:,1]
et_prob = et.predict_proba(today_df)
et_prob = et_prob[:,1]
sv_prob = sv.predict_proba(today_df)
sv_prob = sv_prob[:,1]

# Process predictions for neat display in dataframe
games_data = np.empty((len(games),10))
games_data[:,0] = np.around(rf_prob*100,2)
games_data[:,1] = np.around(norm(0,10.5).ppf(rf_prob),1)    # Convert win probability into point spread
games_data[:,2] = np.around(xgb_prob*100,2)
games_data[:,3] = np.around(norm(0,10.5).ppf(xgb_prob),1)
games_data[:,4] = np.around(gb_prob*100,2)
games_data[:,5] = np.around(norm(0,10.5).ppf(gb_prob),1)
games_data[:,6] = np.around(sv_prob*100,2)
games_data[:,7] = np.around(norm(0,10.5).ppf(sv_prob),1)
games_data[:,8] = np.around(et_prob*100,2)
games_data[:,9] = np.around(norm(0,10.5).ppf(et_prob),1)
games_str = []
for game in games:
    t1 = game[0]
    t2 = game[1]
    teams = t1 + ' at ' + t2
    games_str.append(teams)

# Before calibration:
print("*******************BEFORE CALIBRATION*******************")

games_df = pd.DataFrame(data=games_data,index=games_str,columns=['RF Road Team Win Prob','RF Road Team Point Spread','XGB Road Team Win Prob','XGB Road Team Point Spread','GB Road Team Win Prob','GB Road Team Point Spread','SV Road Team Win Prob','SV Road Team Point Spread','ET Road Team Win Prob','ET Road Team Point Spread'])
print(games_df)


# After calibration
print(" ")
print("*******************CALIBRATED PROBABILITIES*******************")

rf_prob_iso = rf_isotonic.predict_proba(today_df)
rf_prob_iso = rf_prob_iso[:,1]
xgb_prob_sig = xgb_sigmoid.predict_proba(today_df)
xgb_prob_sig = xgb_prob_sig[:,1]
gb_prob_sig = gb_sigmoid.predict_proba(today_df)
gb_prob_sig = gb_prob_sig[:,1]
et_prob_iso = et_isotonic.predict_proba(today_df)
et_prob_iso = et_prob_iso[:,1]
sv_prob_sig = sv_sigmoid.predict_proba(today_df)
sv_prob_sig = sv_prob_sig[:,1]

games_data_cal = np.empty((len(games),10))
games_data_cal[:,0] = np.around(rf_prob*100,2)
games_data_cal[:,1] = np.around(rf_prob_iso*100,2)
games_data_cal[:,2] = np.around(xgb_prob*100,2)
games_data_cal[:,3] = np.around(xgb_prob_sig*100,2)
games_data_cal[:,4] = np.around(gb_prob*100,2)
games_data_cal[:,5] = np.around(gb_prob_sig*100,2)
games_data_cal[:,6] = np.around(sv_prob*100,2)
games_data_cal[:,7] = np.around(sv_prob_sig*100,2)
games_data_cal[:,8] = np.around(et_prob*100,2)
games_data_cal[:,9] = np.around(et_prob_iso*100,2)

games_df_cal = pd.DataFrame(data=games_data_cal,index=games_str,columns=['RF Road Team Win Prob','RF_ISO Road Team Win Prob','XGB Road Team Win Prob','XGB_SIG Road Team Win Prob','GB Road Team Win Prob','GB_SIG Road Team Win Prob','SV Road Team Win Prob','SV_SIG Road Team Win Prob','ET Road Team Win Prob','ET_ISO Road Team Win Prob'])
print(games_df_cal)