import os
import csv
from src.config import LEAGUE_IDS, MAX_WEEKS_BY_YEAR

VALID_MAP_FILE = ".valid_owner_map.csv"
MANUAL_MAP_FILE = ".manual_owner_map.csv"

def load_valid_map():
    valid_map = {}
    if os.path.exists(VALID_MAP_FILE):
        with open(VALID_MAP_FILE, newline='') as f:
            for row in csv.DictReader(f):
                valid_map[row["guid"]] = row["name"]
    return valid_map

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

def get_display_name(guid, team_name, valid_map, manual_map):
    if guid and guid != "--":
        return valid_map.get(guid, "Unknown")
    return manual_map.get(team_name, "Unknown")

def game_result(player_score, opponent_score):
    if not (player_score and opponent_score) or (float(player_score) == 0 and float(opponent_score) == 0):
        return "N/A"
    return "Win" if float(player_score) > float(opponent_score) else "Loss" if float(player_score) < float(opponent_score) else "Tie"

def extract_team_data(team, valid_map, manual_map):
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