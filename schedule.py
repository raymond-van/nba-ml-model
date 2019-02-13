import numpy as np
import pandas as pd
import nba_py
from nba_py import game
from nba_py import constants
from nba_py import team
from nba_py import Scoreboard
import requests
import requests_cache
from datetime import date
pd.set_option('display.max_columns', None)

# Get todays date for input to nba_py module
today = str(date.today())
year, month, day = today.split("-")
year, month, day = int(year), int(month), int(day)

# Returns dataframe about todays NBA games, contains the ID's of the teams that are playing
games_df = Scoreboard(month=month,day=day,year=year)
games_df = games_df.game_header()

games = []

# Process dataframe and returns games which is a list of games being played
for index, row in games_df.iterrows():
    game = []
    # Get team ID and convert to team name
    road_team = row['VISITOR_TEAM_ID']
    road_team = team.TeamSummary(road_team)
    road_team = road_team.info()
    road_team = road_team['TEAM_NAME'].values[0]
    game.append(road_team)
    home_team = row['HOME_TEAM_ID']
    home_team = team.TeamSummary(home_team)
    home_team = home_team.info()
    home_team = home_team['TEAM_NAME'].values[0]
    game.append(home_team)
    games.append(game)

num_of_games = len(games)