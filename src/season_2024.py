from yahoo_oauth import OAuth2
from yahoo_fantasy_api import Game
from src.config import LEAGUE_ID_2024
import pandas as pd
import pprint


oauth = OAuth2(None, None, from_file='oauth2.json')
gm = Game(oauth, 'nfl')

league = gm.to_league(LEAGUE_ID_2024)

raw_data = league.matchups(1)
matchups = raw_data['fantasy_content']['league'][1]