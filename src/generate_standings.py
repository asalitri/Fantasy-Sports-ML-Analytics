import os
import csv
import sys
from yahoo_oauth import OAuth2
from yahoo_fantasy_api import League
from collections import defaultdict
from src.config import LEAGUE_IDS, MAX_WEEKS_BY_YEAR
from src.standings.utils import (
    win_pct,
    is_complete_year,
    playoff_four_teams,
    playoff_six_teams,
    playoff_seven_teams,
    playoff_eight_teams
)

MATCHUP_FILE = "data/matchup_data.csv"
STANDINGS_DIRECTORY = "standings"
ALL_TIME_FILE = "standings_all_time.csv"

os.makedirs(STANDINGS_DIRECTORY, exist_ok=True)

def load_matchups():
    """
    Loads matchup data from the matchup CSV file.

    Returns:
        list[dict]: List of matchups.
    """
    with open(MATCHUP_FILE, newline="") as f:
        return list(csv.DictReader(f))

def calculate_standings(data):
    """
    Compiles win/loss records, points for/against, win percentage, games played, and games back from top ranked team.

    Sorts standings via win percentage and points for as tiebreaker (in decending order). 

    Args:
        data (list[dict]): List of matchup records.

    Returns:
        list[tuple]: Sorted standings as (team_name, stats_dict).
    """
    standings = defaultdict(lambda: {"W": 0, "L": 0, "T": 0, "GP": 0, "Pct": 0.0, "PF": 0.0, "PA": 0.0})  # initializes data to zeros when new player is found, dict of dicts

    for row in data:
        if row["team_1_result"] == "N/A" or row["team_2_result"] == "N/A":  # checks for incomplete match
            continue
        
        team_1 = row["team_1"]
        team_1_result = row["team_1_result"]
        team_2 = row["team_2"]
        
        try:
            team_1_score = float(row["team_1_score"])
            team_2_score = float(row["team_2_score"])
        except (ValueError, TypeError):
            continue

        standings[team_1]["PF"] += team_1_score
        standings[team_1]["PA"] += team_2_score
        standings[team_2]["PF"] += team_2_score
        standings[team_2]["PA"] += team_1_score
        standings[team_1]["GP"] += 1
        standings[team_2]["GP"] += 1

        if team_1_result == "Win":
            standings[team_1]["W"] += 1
            standings[team_2]["L"] += 1
        elif team_1_result == "Loss":
            standings[team_1]["L"] += 1
            standings[team_2]["W"] += 1
        elif team_1_result == "Tie":
            standings[team_1]["T"] += 1
            standings[team_2]["T"] += 1
        
        standings[team_1]["Pct"] = win_pct(standings[team_1]["W"], standings[team_1]["T"], standings[team_1]["GP"])
        standings[team_2]["Pct"] = win_pct(standings[team_2]["W"], standings[team_2]["T"], standings[team_2]["GP"])

    sorted_standings = sorted(  # creates sorted list of tuples, ("name", {stats})
        standings.items(),
        key=lambda team: (team[1]["Pct"], team[1]["PF"]),  # sorts by Pct first, then PF
        reverse=True  # highest to lowest for both Pct and PF
    )

    if not sorted_standings:  # if season is valid, but no games have been played (currently offseason)
        raise ValueError("No data in season yet")

    for rank, (team, stats) in enumerate(sorted_standings, start=1):
        stats["Rank"] = rank

    leader_team, leader_stats = sorted_standings[0]  # leader name and stats (to reference for GB)
    sorted_standings[0][1]["GB"] = 0.0  # sets leader GB to zero
    leader_wins = leader_stats["W"]  # wins for leader
    leader_losses = leader_stats["L"]  # losses for leader
    leader_pct = leader_stats["Pct"]  # win pct for leader

    for team, stats in sorted_standings[1:]:  # calculating GB for rest of teams
        stats["GB"] = ((leader_wins - stats["W"]) + (stats["L"] - leader_losses)) / 2

    return sorted_standings

def get_final_standings(year):
    """
    Computes final standings (after completion of playoffs) for a given year, including playoff and consolation brackets.

    Args:
        year (str or int): Season year.

    Returns:
        list[tuple]: Sorted final standings as (team_name, stats_dict).
    """
    league_id = LEAGUE_IDS[int(year)]
    oauth = OAuth2(None, None, from_file="oauth2.json")
    league = League(oauth, league_id)

    try:
        settings = league.settings()
    except Exception as e:
        print(f"Error getting data from Yahoo for {year}: {e}")
        return None

    matchups = load_matchups()
    num_playoff_teams = int(settings["num_playoff_teams"])
    num_teams = int(settings["num_teams"])
    all_season_matchups = [m for m in matchups if m["year"] == str(year)]
    playoff_matchups = [m for m in all_season_matchups if m["is_playoff"] == "True"]
    reg_season_matchups = [m for m in all_season_matchups if m["is_playoff"] == "False"]

    reg_season_standings = calculate_standings(reg_season_matchups)
    raw_standings = calculate_standings(all_season_matchups)

    playoff_map = {}
    if num_playoff_teams == 6:
        playoff_teams = reg_season_standings[:6]
        playoff_set = {name for name, _ in playoff_teams}
        consolation_teams = reg_season_standings[6:]
        consolation_set = {name for name, _ in consolation_teams}

        playoff_map = playoff_six_teams(playoff_set, playoff_matchups)
        num_consolation_teams = num_teams - 6

        if num_consolation_teams == 4:
            consolation_map = playoff_four_teams(consolation_set, playoff_matchups)
        elif num_consolation_teams == 6:
            consolation_map = playoff_six_teams(consolation_set, playoff_matchups)
        
        for team in consolation_map:  # adjusts final rankings (i.e. 1st in consolation would become 7th place final)
            playoff_map[team] = consolation_map[team] + num_playoff_teams
    elif num_playoff_teams == 7:
        playoff_teams = reg_season_standings[:7]
        playoff_set = {name for name, _ in playoff_teams}
        non_playoff_teams = reg_season_standings[7:]

        playoff_map = playoff_seven_teams(playoff_set, playoff_matchups)

        for team, stats in non_playoff_teams:
            playoff_map[team] = stats["Rank"]
    elif num_playoff_teams == 8:
        playoff_teams = reg_season_standings[:8]
        playoff_set = {name for name, _ in playoff_teams}
        non_playoff_teams = reg_season_standings[8:]

        playoff_map = playoff_eight_teams(playoff_set, playoff_matchups)

        for team, stats in non_playoff_teams:
            playoff_map[team] = stats["Rank"]

    for team, stats in raw_standings:
        stats["Rank"] = playoff_map[team]

    sorted_standings = sorted(raw_standings, key=lambda team: (team[1]["Rank"]))

    return sorted_standings

def save_standings(output_file, data):
    """
    Saves calculated standings to a CSV file. Only used for regular and raw standings.

    Args:
        output_file (str): Output CSV filepath.
        data (list[dict]): List of matchup records.
    """
    try:
        standings = calculate_standings(data)
    except ValueError as e:
        raise

    fieldnames = ["Rank", "Player", "W", "L", "T", "GP", "Pct", "PF", "PA", "GB"]
    with open(output_file, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for name, stats in standings:
            row = {"Player": name}
            row.update(stats)
            writer.writerow(row)

def save_final_standings(output_file, year):
    """
    Saves final standings for a given year to a CSV file.

    Args:
        output_file (str): Output CSV filepath.
        year (str or int): Season year.
    """
    try:
        standings = get_final_standings(year)
    except Exception as e:
        raise

    fieldnames = ["Rank", "Player", "W", "L", "T", "GP", "Pct", "PF", "PA", "GB"]
    with open(output_file, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for name, stats in standings:
            row = {"Player": name}
            row.update(stats)
            writer.writerow(row)

def update_season_standings(year):
    """
    Updates all applicable standings files for a given season (year). Prints which files have been successfully created.

    Args:
        year (str or int): Season year.
    """
    if int(year) not in LEAGUE_IDS:
        raise ValueError(f"Invalid year: {year}")
    matchups = load_matchups()
    reg_season_matchups = [m for m in matchups if m["year"] == str(year) and m["is_playoff"] == "False"]
    all_season_matchups = [m for m in matchups if m["year"] == str(year)]
    os.makedirs(f"{STANDINGS_DIRECTORY}/{year}", exist_ok=True)
    reg_filename = f"{STANDINGS_DIRECTORY}/{year}/regular.csv"
    raw_filename = f"{STANDINGS_DIRECTORY}/{year}/raw.csv"
    final_filename = f"{STANDINGS_DIRECTORY}/{year}/final.csv"
    try:
        if is_complete_year(year, matchups):
            save_final_standings(final_filename, year)
        save_standings(reg_filename, reg_season_matchups)
        print(f"\n{year} regular season standings successfully updated in {reg_filename}.")
        save_standings(raw_filename, all_season_matchups)
        print(f"{year} raw standings successfully updated in {raw_filename}.")
        if is_complete_year(year, matchups):
            print(f"{year} final standings successfully updated in {final_filename}.")
    except ValueError as e:
        if os.path.isdir(f"{STANDINGS_DIRECTORY}/{year}") and not os.listdir(f"{STANDINGS_DIRECTORY}/{year}"):
            os.rmdir(f"{STANDINGS_DIRECTORY}/{year}")
        print(f"{e}: {year}")

def update_all_time_standings():
    """
    Updates and saves the all-time standings across all seasons. Prints on success.
    """
    matchups = load_matchups()
    filename = f"{STANDINGS_DIRECTORY}/all_time.csv"
    save_standings(filename, matchups)
    print(f"\nAll time standings successfully updated in {filename}.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m src.standings season <year>")
        print("  python -m src.standings all_time")
        sys.exit(1)

    command = sys.argv[1]

    if command == "season":
        if len(sys.argv) != 3:
            print("Usage: python3 -m src.standings season <year>")
            sys.exit(1)
        year = sys.argv[2]
        try:
            update_season_standings(year)
        except ValueError as e:
            print(e)
            sys.exit(1)
    elif command == "all_time":
        update_all_time_standings()
    else:
        print("Unknown command. Use 'season' or 'all_time'.")
        sys.exit(1)