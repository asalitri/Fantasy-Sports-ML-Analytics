import os
import csv
import sys
from yahoo_oauth import OAuth2
from yahoo_fantasy_api import League
from collections import defaultdict
from src.config import LEAGUE_IDS, MAX_WEEKS_BY_YEAR
from src.matchup_utils import (
    load_valid_map,
    load_manual_map,
    get_display_name
)
from pprint import pprint


MATCHUP_FILE = "data/matchup_data.csv"
STANDINGS_DIRECTORY = "standings"
ALL_TIME_FILE = "standings_all_time.csv"

os.makedirs(STANDINGS_DIRECTORY, exist_ok=True)

def win_pct(wins, ties, gp):
    return (wins + 0.5 * ties) / gp if gp > 0 else 0.0

def calculate_standings(data):
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
    league_id = LEAGUE_IDS[int(year)]
    oauth = OAuth2(None, None, from_file="oauth2.json")
    league = League(oauth, league_id)

    try:
        standings_data = league.standings()
        teams = league.teams()
    except Exception as e:
        print(f"Error getting data from Yahoo for {year}: {e}")
        return None
    
    valid_map = load_valid_map()
    manual_map = load_manual_map()
    standings = []
    team_key_to_guid = {}
    for team in teams.values():
        team_key = team["team_key"]
        guid = team["managers"][0]["manager"]["guid"]

        if team_key and guid:
            team_key_to_guid[team_key] = guid
    for team in standings_data:
        team_key = team["team_key"]
        team_name = team["name"]
        guid = team_key_to_guid.get(team_key, None)
        name = get_display_name(guid, team_name, valid_map, manual_map)

        outcomes = team.get('outcome_totals', {})
        pf = float(team.get('points_for', 0.0))
        pa = float(team.get('points_against', 0.0))
        wins = int(outcomes.get('wins', 0))
        losses = int(outcomes.get('losses', 0))
        ties = int(outcomes.get('ties', 0))
        gp = wins + losses + ties
        pct = win_pct(wins, ties, gp)

        standings.append({
            "Player": name,
            "W": wins,
            "L": losses,
            "T": ties,
            "GP": gp,
            "Pct": pct,
            "PF": pf,
            "PA": pa,
        })

    sorted_standings = sorted(  # creates sorted list of dicts
        standings,
        key=lambda team: (team["Pct"], team["PF"]),  # sorts by Pct first, then PF
        reverse=True  # highest to lowest for both Pct and PF
    )

    for rank, sorted_team in enumerate(sorted_standings, start=1):
        sorted_team["Rank"] = rank

    return sorted_standings

def save_standings(output_file, data):
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
    try:
        standings = get_final_standings(year)
    except Exception as e:
        raise

    fieldnames = ["Rank", "Player", "W", "L", "T", "GP", "Pct", "PF", "PA"]
    with open(output_file, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for team in standings:
            writer.writerow(team)

def load_matchups():  # return list of dicts for matchup data
    with open(MATCHUP_FILE, newline="") as f:
        return list(csv.DictReader(f))

def is_complete_year(year, matchups):  # returns boolean if the given season is complete
    final_week = MAX_WEEKS_BY_YEAR[int(year)]
    final_week_matchups = [
        m for m in matchups
        if int(m["year"]) == int(year) and int(m["week"]) == final_week
    ]

    for m in final_week_matchups:
        if (
            m["team_1_result"] == "N/A" or
            m["team_2_result"] == "N/A" or
            not m["team_1_score"] or
            not m["team_2_score"]
        ):
            return False

    return bool(final_week_matchups)

def update_season_standings(year):
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
        print(f"{e}: {year}")

def update_all_time_standings():
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