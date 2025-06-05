import os
import csv
from yahoo_oauth import OAuth2
from yahoo_fantasy_api import Game
from src.config import LEAGUE_IDS, MAX_WEEKS_BY_YEAR

VALID_MAP_FILE = ".valid_owner_map.csv"
MANUAL_MAP_FILE = "manual_owner_map.csv"
OUTPUT_FILE = "data/matchup_data.csv"

# returned as: {guid: name}
# given file as: name,guid,yahoo_nickname
def load_valid_map():
    valid_map = {}
    if os.path.exists(VALID_MAP_FILE):
        with open(VALID_MAP_FILE, newline='') as f:
            for row in csv.DictReader(f):
                valid_map[row["guid"]] = row["name"]
    return valid_map

# returned as: {team_name: name}
# given file as: name,manual_id,team1,team2,team3
def load_manual_map():
    manual_map = {}
    if os.path.exists(MANUAL_MAP_FILE):
        with open(MANUAL_MAP_FILE, newline='') as f:
            for row in csv.DictReader(f):
                name = row['name']
                for col in ['team1', 'team2', 'team3']:
                    team_name = row.get(col)
                    if team_name:
                        manual_map[team_name] = name
    return manual_map

# grabs display name using either guid (if applicable) or team_name as fallback
def get_display_name(guid, team_name, valid_map, manual_map):
    if guid and guid != "--":
        return valid_map.get(guid, "Unknown")
    return manual_map.get(team_name, "Unknown")

# finds result of game (for player_score)
# returned as: "Win", "Loss", "Tie", or "N/A"
def game_result(player_score, opponent_score):
    if not (player_score and opponent_score) or (float(player_score) == 0 and float(opponent_score) == 0):
        return "N/A"
    return "Win" if float(player_score) > float(opponent_score) else "Loss" if float(player_score) < float(opponent_score) else "Tie"

# main compilation of matchup data
def collect_matchups():
    print("beginning collect_matchups()")

    oauth = OAuth2(None, None, from_file="oauth2.json")
    gm = Game(oauth, "nfl")
    valid_map = load_valid_map()
    manual_map = load_manual_map()
    all_matchups = []

    print("starting loop in collect_matchups()")

    for year, league_id in LEAGUE_IDS.items():
        league = gm.to_league(league_id)

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

                    # extracts display name, week score, and week projection for given team
                    def extract_team_data(team):
                        meta = team[0]  # meta data about team
                        stats = team[1]  # matchup info about team
                        team_name = ""
                        guid = "--"
                        for item in meta:
                            if isinstance(item, dict):
                                if "name" in item:
                                    team_name = item["name"]
                                if "managers" in item:
                                    managers = item["managers"]
                                    if managers and "guid" in managers[0]["manager"]:  # only checking first manager if multiple managers
                                        guid = managers[0]["manager"]["guid"]
                        actual = stats.get("team_points", {}).get("total")
                        projected = stats.get("team_projected_points", {}).get("total")
                        return get_display_name(guid, team_name, valid_map, manual_map), actual, projected

                    team_1_display, team_1_score, team_1_proj = extract_team_data(team_1)
                    team_2_display, team_2_score, team_2_proj = extract_team_data(team_2)
                    team_1_result = game_result(team_1_score, team_2_score)
                    team_2_result = game_result(team_2_score, team_1_score)

                    all_matchups.append({
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
            except Exception as e:
                print(f"Error in {year}, Week {week}: {e}")
                continue
    print("done with collect_matchups()")

    return all_matchups

# saves matchup data to csv file given list of matchup dictionaries (from collect_matchups())
def save_to_csv(matchups):
    print("beginning save_to_csv()")

    os.makedirs("data", exist_ok=True)
    fieldnames = [
        "year", "week", "team_1", "team_1_score", "team_1_proj", "team_1_result",
        "team_2", "team_2_score", "team_2_proj", "team_2_result"
    ]
    with open(OUTPUT_FILE, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in matchups:
            writer.writerow(row)
    print(f"\nMatchups successfully saved to {OUTPUT_FILE}.")
                    
if __name__ == "__main__":
    matchups = collect_matchups()
    save_to_csv(matchups)