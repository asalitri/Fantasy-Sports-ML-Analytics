import os
import csv
from src.config import LEAGUE_IDS, MAX_WEEKS_BY_YEAR

VALID_MAP_FILE = ".valid_owner_map.csv"
MANUAL_MAP_FILE = "manual_owner_map.csv"

def load_valid_map():
    """
    Loads a mapping of Yahoo manager GUIDs to display names from the valid owner CSV file.

    Given CSV file as: name,guid,yahoo_nickname.

    Returns:
        dict: {guid: name}
    """
    valid_map = {}
    if os.path.exists(VALID_MAP_FILE):
        with open(VALID_MAP_FILE, newline='') as f:
            for row in csv.DictReader(f):
                valid_map[row["guid"]] = row["name"]
    return valid_map

def load_manual_map():
    """
    Loads a manual mapping of team names to display names from a CSV file.

    Given CSV file as: name,manual_id,team1,team2,team3.

    Returns:
        dict: {team_name: name}
    """
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
    """
    Grabs display name using either guid (if applicable) or team_name as fallback.

    Args:
        guid (str): Yahoo manager GUID.
        team_name (str): Raw team name from the API.
        valid_map (dict): Mapping from guid to display name.
        manual_map (dict): Mapping from team name to manual display name.

    Returns:
        str: Display name for the team.
    """
    if guid and guid != "--":
        return valid_map.get(guid, "Unknown")
    return manual_map.get(team_name, "Unknown")

# finds result of game (for player_score)
# returned as: "Win", "Loss", "Tie", or "N/A"
def game_result(player_score, opponent_score):
    """
    Finds result of game (for player_score).

    Args:
        player_score (float or str): The score of the player.
        opponent_score (float or str): The score of the opponent.

    Returns:
        str: One of "Win", "Loss", "Tie", or "N/A" (if game is yet to be played).
    """
    if not (player_score and opponent_score) or (float(player_score) == 0 and float(opponent_score) == 0):
        return "N/A"
    return "Win" if float(player_score) > float(opponent_score) else "Loss" if float(player_score) < float(opponent_score) else "Tie"

def extract_team_data(team, valid_map, manual_map):
    """
    Extracts and returns display name, actual score, and projected score for a fantasy team.

    Args:
        team (dict): Raw team data from Yahoo API.
        valid_map (dict): Mapping from guid to display name.
        manual_map (dict): Mapping from team name to manual display name.

    Returns:
        tuple: (display_name, actual_score, projected_score).
    """
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