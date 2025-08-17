import os
import csv
import sys
from yahoo_oauth import OAuth2
from yahoo_fantasy_api import League
from src.config import LEAGUE_IDS, MAX_WEEKS_BY_YEAR
from src.matchup_utils import (
    load_valid_map,
    load_manual_map,
    game_result,
    extract_team_data
)

OUTPUT_FILE = "data/matchup_data.csv"

# main compilation of matchup data
def collect_matchups():
    oauth = OAuth2(None, None, from_file="oauth2.json")
    valid_map = load_valid_map()
    manual_map = load_manual_map()
    all_matchups = []

    for year, league_id in LEAGUE_IDS.items():
        league = League(oauth, league_id)

        max_week = MAX_WEEKS_BY_YEAR.get(year, 17)  # defaulted to 17 for future years if year not listed
        for week in range(1, max_week + 1):
            try:
                raw_data = league.matchups(week)
                matchups = raw_data['fantasy_content']['league'][1]['scoreboard']['0']['matchups']

                for matchup in matchups.values():
                    if not (isinstance(matchup, dict) and "matchup" in matchup):
                        continue

                    teams = matchup['matchup']['0']['teams']
                    team_1 = teams['0']['team']
                    team_2 = teams['1']['team']

                    team_1_display, team_1_score, team_1_proj = extract_team_data(team_1, valid_map, manual_map)
                    team_2_display, team_2_score, team_2_proj = extract_team_data(team_2, valid_map, manual_map)
                    team_1_result = game_result(team_1_score, team_2_score)
                    team_2_result = game_result(team_2_score, team_1_score)

                    is_playoff = (max_week - week < 3)  # if matchup is a playoff game

                    all_matchups.append({
                        "year": year,
                        "week": week,
                        "is_playoff": is_playoff,
                        "team_1": team_1_display,
                        "team_1_score": team_1_score,
                        "team_1_proj": team_1_proj,
                        "team_1_result": team_1_result,
                        "team_2": team_2_display,
                        "team_2_score": team_2_score,
                        "team_2_proj": team_2_proj,
                        "team_2_result": team_2_result,
                    })
            except Exception as e:
                print(f"Error in {year}, Week {week}: {e}")
                continue

    return all_matchups

def save_to_csv(matchups):
    os.makedirs("data", exist_ok=True)
    fieldnames = [
        "year", "week", "is_playoff", "team_1", "team_1_score", "team_1_proj", "team_1_result",
        "team_2", "team_2_score", "team_2_proj", "team_2_result"
    ]
    with open(OUTPUT_FILE, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in matchups:
            writer.writerow(row)
    print(f"\nMatchups successfully saved to {OUTPUT_FILE}.")

def main():
    try:
        matchups = collect_matchups()
        save_to_csv(matchups)
    except Exception as e:
        print(f"Error collecting matchups: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()