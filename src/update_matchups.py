import os
import csv
import sys
from yahoo_oauth import OAuth2
from yahoo_fantasy_api import Game
from src.config import LEAGUE_IDS, MAX_WEEKS_BY_YEAR
from src.matchup_utils import (
    load_valid_map,
    load_manual_map,
    game_result,
    extract_team_data
)

OUTPUT_FILE = "data/matchup_data.csv"

def get_week_matchups(year, week, oauth, valid_map, manual_map):
    league_id = LEAGUE_IDS.get(year)
    if not league_id:
        raise ValueError(f"Invalid year: {year}")
    
    max_week = MAX_WEEKS_BY_YEAR.get(year, 17)  # defaulted to 17 for future years if year not listed
    if not (1 <= week <= max_week):
        raise ValueError(f"Invalid week: {week} for {year} season. Valid weeks are 1 to {max_week}.")

    league = Game(oauth, "nfl").to_league(league_id)
    raw_data = league.matchups(week)
    matchups = raw_data['fantasy_content']['league'][1]['scoreboard']['0']['matchups']
    week_data = []

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

        week_data.append({
            "year": year,
            "week": week,
            "team_1": team_1_display,
            "team_1_score": team_1_score,
            "team_1_proj": team_1_proj,
            "team_1_result": team_1_result,
            "team_2": team_2_display,
            "team_2_score": team_2_score,
            "team_2_proj": team_2_proj,
            "team_2_result": team_2_result,
        })

    return week_data
    
def update_week_csv(year, week):
    oauth = OAuth2(None, None, from_file="oauth2.json")
    valid_map = load_valid_map()
    manual_map = load_manual_map()
    new_week_data = get_week_matchups(year, week, oauth, valid_map, manual_map)

    # makes sure matchups file already exists
    if not os.path.exists(OUTPUT_FILE):
        raise FileNotFoundError(f"{OUTPUT_FILE} not found. Run collect_matchups().")

    with open(OUTPUT_FILE, newline='') as f:
        reader = csv.DictReader(f)
        existing_data = list(reader)
        fieldnames = reader.fieldnames

    lookup = {
        (int(row["year"]), int(row["week"]), row["team_1"], row["team_2"]): row
        for row in new_week_data
    }

    for row in existing_data:
        key = (int(row["year"]), int(row["week"]), row["team_1"], row["team_2"])
        if key in lookup:
            row.update(lookup[key])

    with open(OUTPUT_FILE, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in existing_data:
            writer.writerow(row)
    print(f"\nMatchups from {year} week {week} successfully saved to {OUTPUT_FILE}.")

if __name__ == "__main__":
    if len(sys.argv) != 3:  # checks for exactly two args (year, week) in addition to calling for the .py file
        print("Usage: python3 -m src.update_matchups <year> <week>")
        sys.exit(1)
    
    try:
        year = int(sys.argv[1])
        week = int(sys.argv[2])
    except ValueError:
        print("Error: <year> and <week> parameters must be integers.")
        print("Usage: python3 -m src.update_matchups <year> <week>")
        sys.exit(1)
    try:
        update_week_csv(year, week)
    except ValueError as e:
        print(e)
        sys.exit(1)