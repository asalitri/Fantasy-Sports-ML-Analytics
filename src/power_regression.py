import csv
import os
from src.config import MATCHUP_FILE
from src.generate_stats import calculate_stats
from src.generate_standings import calculate_standings
from src.power_utils import 

def load_matchups():
    """
    Loads matchup data from the matchup CSV file.

    Returns:
        list[dict]: List of matchups.
    """
    with open(MATCHUP_FILE, newline="") as f:
        return list(csv.DictReader(f))

def extract_features_and_target(matchups, year, week):
    matchups_up_to_week = [m for m in matchups if int(m["week"]) <= week]
    reg_season_matchups = [m for m in matchups if m["year"] == str(year) and m["is_playoff"] == "False"]

    current_stats = calculate_stats(matchups_up_to_week, year)
    current_standings = calculate_standings(matchups_up_to_week)
    final_reg_standings = calculate_standings(reg_season_matchups)

    # lookup tables
    current_wins = {row["Player"]: int(row["W"]) for row in current_standings}
    final_wins = {row["player"]: int(row["W"]) for row in final_reg_standings}

    win_pcts = []
    ewmas = []
    stdevs = []
    win_streaks = []

    rows = []
    for stats_row, standings_row in zip(current_stats, current_standings):
        player = stats_row["Player"]
        try:
            win_pct = float(standings_row["Pct"])
            ewma = float(stats_row["EWMA_avg"])
            stdev = float(stats_row["StDev"])
            win_streak = float(stats_row["WinStreak"])

            win_pcts.append(win_pct)
            ewmas.append(ewma)
            stdevs.append(stdev)
            win_streaks.append(win_streak)
        except ValueError:
            continue

    







