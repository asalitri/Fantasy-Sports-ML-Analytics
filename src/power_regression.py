import csv
import os
from src.config import MATCHUP_FILE
from src.generate_stats import calculate_stats, add_ewma
from src.generate_standings import calculate_standings
from src.power_utils import scaled_metric
from src.config import LEAGUE_IDS, MAX_WEEKS_BY_YEAR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import math


def load_matchups():
    """
    Loads matchup data from the matchup CSV file.

    Returns:
        list[dict]: List of matchups.
    """
    with open(MATCHUP_FILE, newline="") as f:
        return list(csv.DictReader(f))

def extract_features_and_target(matchups, year, week):
    matchups_up_to_week = [m for m in matchups if m["year"] == str(year) and int(m["week"]) <= week]
    reg_season_matchups = [m for m in matchups if m["year"] == str(year) and m["is_playoff"] == "False"]

    current_stats_pre_ewma = calculate_stats(matchups_up_to_week, year)
    current_stats = add_ewma(year, current_stats_pre_ewma)
    current_standings = calculate_standings(matchups_up_to_week)
    final_reg_standings = calculate_standings(reg_season_matchups)

    # lookup tables
    current_wins = {row[0]: int(row[1]["W"]) for row in current_standings}
    final_wins = {row[0]: int(row[1]["W"]) for row in final_reg_standings}

    win_pcts = []
    ewmas = []
    stdevs = []
    streaks = []
    future_wins_target = []

    rows = []
    for stats_row, standings_row in zip(current_stats, current_standings):
        player = stats_row["Player"]
        try:
            win_pct = float(standings_row[1]["Pct"])
            ewma = float(stats_row["EWMA"])
            stdev = float(stats_row["StDev"])
            win_streak = float(stats_row["Streak"])

            win_pcts.append(win_pct)
            ewmas.append(ewma)
            stdevs.append(stdev)
            streaks.append(win_streak)

            future_wins = final_wins[player] - current_wins[player]
            future_wins_target.append(future_wins)
        except ValueError:
            continue

    # local scaling
    scaled_ewmas = scaled_metric(ewmas)
    scaled_stdevs = scaled_metric(stdevs)

    return {
        "ewma": scaled_ewmas,
        "stdev": scaled_stdevs,
        "win_pct": win_pcts,  # unscaled
        "streak": streaks,  # unscaled
        "target": future_wins_target
    }

def aggregate_features_and_targets(matchups, week):
    all_scaled_ewmas = []
    all_scaled_stdevs = []
    all_win_pcts = []
    all_streaks = []
    all_targets = []

    for year in list(LEAGUE_IDS)[:-1]:  # loops thru completed years
        result = extract_features_and_target(matchups, year, week)

        all_scaled_ewmas.extend(result["ewma"])
        all_scaled_stdevs.extend(result["stdev"])
        all_win_pcts.extend(result["win_pct"])
        all_streaks.extend(result["streak"])
        all_targets.extend(result["target"])

    # global scaling
    all_scaled_win_pcts = scaled_metric(all_win_pcts)
    all_scaled_streaks = scaled_metric(all_streaks)

    X = []
    for i in range(len(all_targets)):
        features = [
            all_scaled_ewmas[i],
            all_scaled_stdevs[i],
            all_scaled_win_pcts[i],
            all_scaled_streaks[i]
        ]
        X.append(features)

    y = all_targets

    return X, y

def run_weekly_regression(X, y):
    model = LinearRegression()
    model.fit(X, y)

    y_prediction = model.predict(X)

    r2 = r2_score(y, y_prediction)
    rmse = math.sqrt(mean_squared_error(y, y_prediction))

    return {
        "coefficients": model.coef_.tolist(),
        "intercept": model.intercept_,
        "r2": r2,
        "rmse": rmse
    }

def main():
    matchups = load_matchups()

    for week in range(3, 12):  # Example range
        X, y = aggregate_features_and_targets(matchups, week)
        result = run_weekly_regression(X, y)

        print(f"Week {week}")
        print("  Coefficients:", result["coefficients"])
        print("  Intercept:", result["intercept"])
        print("  R²:", round(result["r2"], 3))
        print("  RMSE:", round(result["rmse"], 3))
        print()

if __name__ == "__main__":
    main()
    







