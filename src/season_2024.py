from yahoo_oauth import OAuth2
from yahoo_fantasy_api import Game
from src.config import LEAGUE_ID_2024
import pandas as pd
import os


oauth = OAuth2(None, None, from_file='oauth2.json')
gm = Game(oauth, 'nfl')
league = gm.to_league(LEAGUE_ID_2024)

matchup_data = []

for week in range(1, 18):
    raw_data = league.matchups(week)
    matchups = raw_data['fantasy_content']['league'][1]['scoreboard']['0']['matchups']

    for matchup in matchups.values():
        if isinstance(matchup, dict) and 'matchup' in matchup:
            teams = matchup['matchup']['0']['teams']
            team_1 = teams['0']['team']
            team_2 = teams['1']['team']

            team_1_name = None
            for item in team_1[0]:
                if isinstance(item, dict) and 'name' in item:
                    team_1_name = item['name']
                    break

            team_2_name = None
            for item in team_2[0]:
                if isinstance(item, dict) and 'name' in item:
                    team_2_name = item['name']
                    break

            team_1_score = team_1[1]['team_points']['total'] if len(team_1) > 1 and 'team_points' in team_1[1] else None
            team_2_score = team_2[1]['team_points']['total'] if len(team_2) > 1 and 'team_points' in team_2[1] else None

            matchup_data.append({
                'week': week,
                'team_1': team_1_name,
                'team_1_score': team_1_score,
                'team_2': team_2_name,
                'team_2_score': team_2_score
            })

os.makedirs('data', exist_ok=True)
df = pd.DataFrame(matchup_data)
df.to_csv('data/matchups_2024.csv', index=False)
print("successful")