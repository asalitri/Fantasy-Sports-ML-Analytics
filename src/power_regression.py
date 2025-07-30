import csv
import os
from src.config import MATCHUP_FILE
from src.generate_stats import calculate_stats, add_ewma, add_adjusted_avg, add_luck_index
from src.generate_standings import calculate_standings
from src.power_utils import scaled_metric
from src.config import LEAGUE_IDS, MAX_WEEKS_BY_YEAR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
import math
import random
import numpy as np


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

    current_stats = calculate_stats(matchups_up_to_week, year)  # no ewma, adj_avg, luck_index
    current_stats = add_ewma(year, current_stats)
    current_stats = add_adjusted_avg(year, current_stats)
    current_stats = add_luck_index(year, current_stats, matchups)

    current_standings = calculate_standings(matchups_up_to_week)
    final_reg_standings = calculate_standings(reg_season_matchups)

    # lookup tables
    current_wins = {row[0]: int(row[1]["W"]) for row in current_standings}
    final_wins = {row[0]: int(row[1]["W"]) for row in final_reg_standings}

    win_pcts = []
    ewmas = []
    stdevs = []
    streaks = []
    adj_avgs = []
    avgs = []
    pfs = []
    pas = []
    highs = []
    lows = []
    luck_indexes = []
    ewma_deltas = []
    adj_deltas = []
    pf_deltas = []
    range_deltas = []
    range_deltas_2 = []  # range_deltas squared
    interaction_1_list = []  # pf_deltas * range_deltas
    future_wins_target = []

    rows = []
    for stats_row, standings_row in zip(current_stats, current_standings):
        player = stats_row["Player"]
        try:
            win_pct = float(standings_row[1]["Pct"])
            ewma = float(stats_row["EWMA"])
            stdev = float(stats_row["StDev"])
            win_streak = float(stats_row["Streak"])
            
            adj_avg = float(stats_row["AdjustedAvg"])
            avg = float(stats_row["Avg"])
            pf = float(standings_row[1]["PF"])
            pa = float(standings_row[1]["PA"])
            high = float(stats_row["High"])
            low = float(stats_row["Low"])
            luck_index = float(stats_row["LuckIndex"])
            ewma_delta = ewma - avg
            adj_delta = adj_avg - avg
            pf_delta = pf - pa
            range_delta = high - low
            range_delta_2 = range_delta ** 2
            interaction_1 = pf_delta * range_delta

            win_pcts.append(win_pct)
            ewmas.append(ewma)
            stdevs.append(stdev)
            streaks.append(win_streak)

            adj_avgs.append(adj_avg)
            avgs.append(avg)
            pfs.append(pf)
            pas.append(pa)
            highs.append(high)
            lows.append(low)
            luck_indexes.append(luck_index)
            ewma_deltas.append(ewma_delta)
            adj_deltas.append(adj_delta)
            pf_deltas.append(pf_delta)
            range_deltas.append(range_delta)
            range_deltas_2.append(range_delta_2)
            interaction_1_list.append(interaction_1)

            future_wins = final_wins[player] - current_wins[player]
            future_wins_target.append(future_wins)
        except ValueError:
            continue

    # local scaling
    scaled_ewmas = scaled_metric(ewmas)
    scaled_stdevs = scaled_metric(stdevs)

    scaled_adj_avgs = scaled_metric(adj_avgs)
    scaled_avgs = scaled_metric(avgs)
    scaled_pfs = scaled_metric(pfs)
    scaled_pas = scaled_metric(pas)
    scaled_highs = scaled_metric(highs)
    scaled_lows = scaled_metric(lows)
    scaled_ewma_deltas = scaled_metric(ewma_deltas)
    scaled_adj_deltas = scaled_metric(adj_deltas)
    scaled_pf_deltas = scaled_metric(pf_deltas)
    scaled_range_deltas = scaled_metric(range_deltas)
    scaled_range_deltas_2 = scaled_metric(range_deltas_2)
    scaled_interaction_1_list = scaled_metric(interaction_1_list)

    return {
        "ewma": scaled_ewmas,
        "stdev": scaled_stdevs,
        "win_pct": win_pcts,  # unscaled
        "streak": streaks,  # unscaled
        "adj_avg": scaled_adj_avgs,
        "avg": scaled_avgs,
        "pf": scaled_pfs,
        "pa": scaled_pas,
        "high": scaled_highs,
        "low": scaled_lows,
        "luck_index": luck_indexes,
        "ewma_delta": scaled_ewma_deltas,
        "adj_delta": scaled_adj_deltas,
        "pf_delta": scaled_pf_deltas,
        "range_delta": scaled_range_deltas,
        "range_delta_2": scaled_range_deltas_2,
        "interaction_1": scaled_interaction_1_list,
        "target": future_wins_target
    }

def aggregate_features_and_targets(matchups, week, years):
    all_scaled_ewmas = []
    all_scaled_stdevs = []
    all_win_pcts = []
    all_streaks = []

    all_scaled_adj_avgs = []
    all_scaled_avgs = []
    all_scaled_pfs = []
    all_scaled_pas = []
    all_scaled_highs = []
    all_scaled_lows = []
    all_luck_indexes = []
    all_scaled_ewma_deltas = []
    all_scaled_adj_deltas = []
    all_scaled_pf_deltas = []
    all_scaled_range_deltas = []
    all_scaled_range_deltas_2 = []
    all_scaled_interaction_1_list = []

    all_targets = []

    for year in years:  # loops thru completed years
        result = extract_features_and_target(matchups, year, week)

        all_scaled_ewmas.extend(result["ewma"])
        all_scaled_stdevs.extend(result["stdev"])
        all_win_pcts.extend(result["win_pct"])
        all_streaks.extend(result["streak"])

        all_scaled_adj_avgs.extend(result["adj_avg"])
        all_scaled_avgs.extend(result["avg"])
        all_scaled_pfs.extend(result["pf"])
        all_scaled_pas.extend(result["pa"])
        all_scaled_highs.extend(result["high"])
        all_scaled_lows.extend(result["low"])
        all_luck_indexes.extend(result["luck_index"])
        all_scaled_ewma_deltas.extend(result["ewma_delta"])
        all_scaled_adj_deltas.extend(result["adj_delta"])
        all_scaled_pf_deltas.extend(result["pf_delta"])
        all_scaled_range_deltas.extend(result["range_delta"])
        all_scaled_range_deltas_2.extend(result["range_delta_2"])
        all_scaled_interaction_1_list.extend(result["interaction_1"])

        all_targets.extend(result["target"])

    # global scaling
    all_scaled_win_pcts = scaled_metric(all_win_pcts)
    all_scaled_streaks = scaled_metric(all_streaks)
    all_scaled_luck_indexes = scaled_metric(all_luck_indexes)

    X = []
    for i in range(len(all_targets)):
        features = [
            # all_scaled_ewmas[i],
            # all_scaled_stdevs[i],
            # all_scaled_win_pcts[i],
            # all_scaled_streaks[i],
            # all_scaled_adj_avgs[i],
            all_scaled_avgs[i],
            all_scaled_pfs[i],
            all_scaled_pas[i],
            all_scaled_highs[i],
            all_scaled_lows[i],
            all_scaled_luck_indexes[i],
            all_scaled_ewma_deltas[i],
            all_scaled_adj_deltas[i],
            all_scaled_pf_deltas[i],
            all_scaled_range_deltas[i],
            # all_scaled_range_deltas_2[i],
            all_scaled_interaction_1_list[i]
        ]
        X.append(features)

    y = all_targets

    return X, y

def run_forest_regression(matchups, week):
    test_year = random.choice(list(LEAGUE_IDS)[:-1])
    train_years = [year for year in list(LEAGUE_IDS)[:-1] if year != test_year]
    test_years = [test_year]

    X_train, y_train = aggregate_features_and_targets(matchups, week, train_years)
    X_test, y_test = aggregate_features_and_targets(matchups, week, test_years)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_prediction = model.predict(X_test)

    r2 = r2_score(y_test, y_prediction)
    rmse = math.sqrt(mean_squared_error(y_test, y_prediction))

    return {
    "feature_importances": model.feature_importances_.tolist(),
    "r2": r2,
    "rmse": rmse,
    "test_year": test_year  # Might be useful to know which year was used for testing
}

def cross_validation_forest_regression(matchups, week, years):
    r2_scores = []
    rmses = []
    feature_importances = []

    for test_year in years:
        train_years = [y for y in years if y != test_year]

        X_train, y_train = aggregate_features_and_targets(matchups, week, train_years)
        X_test, y_test = aggregate_features_and_targets(matchups, week, [test_year])

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2_scores.append(r2_score(y_test, y_pred))
        rmses.append(np.sqrt(mean_squared_error(y_test, y_pred)))

        feature_importances.append(model.feature_importances_)

    avg_feature_importance = np.mean(feature_importances, axis=0).tolist()

    return {
        "avg_r2": np.mean(r2_scores),
        "avg_rmse": np.mean(rmses),
        "r2_scores": r2_scores,
        "rmses": rmses,
        "avg_feature_importance": avg_feature_importance
    }

def run_linear_regression(X, y):
    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)

    r2 = r2_score(y, y_pred)
    rmse = math.sqrt(mean_squared_error(y, y_pred))

    return {
        "coefficients": model.coef_.tolist(),
        "intercept": model.intercept_,
        "r2": r2,
        "rmse": rmse
    }

def cross_validation_linear_regression(matchups, week, years):
    r2_scores = []
    rmses = []

    for test_year in years:
        train_years = [y for y in years if y != test_year]

        X_train, y_train = aggregate_features_and_targets(matchups, week, train_years)
        X_test, y_test = aggregate_features_and_targets(matchups, week, [test_year])

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2_scores.append(r2_score(y_test, y_pred))
        rmses.append(np.sqrt(mean_squared_error(y_test, y_pred)))

    return {
        "avg_r2": np.mean(r2_scores),
        "avg_rmse": np.mean(rmses),
        "r2_scores": r2_scores,
        "rmses": rmses
    }

def xgb_regression(matchups, week, years):
    r2_scores = []
    rmses = []
    feature_importances = []

    for test_year in years:
        train_years = [y for y in years if y != test_year]

        X_train, y_train = aggregate_features_and_targets(matchups, week, train_years)
        X_test, y_test = aggregate_features_and_targets(matchups, week, [test_year])

        model = XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2_scores.append(r2_score(y_test, y_pred))
        rmses.append(np.sqrt(mean_squared_error(y_test, y_pred)))

        feature_importances.append(model.feature_importances_)

    avg_feature_importance = np.mean(feature_importances, axis=0).tolist()

    return {
        "avg_r2": np.mean(r2_scores),
        "avg_rmse": np.mean(rmses),
        "r2_scores": r2_scores,
        "rmses": rmses,
        "avg_feature_importance": avg_feature_importance
    }

def main():
    matchups = load_matchups()

    for week in range(3, 12):  # Example range
        # X, y = aggregate_features_and_targets(matchups, week)
        # result = run_linear_regression(X, y)
        result = xgb_regression(matchups, week, list(LEAGUE_IDS)[:-1])

        print(f"Week {week}")
        # print("  Test Year:", result["test_year"])
        # print("  Coefficients:", result["coefficients"])
        # print("  Intercept:", result["intercept"])
        # print("  Feature Importances:", result["feature_importances"])
        print("  R²:", round(result["avg_r2"], 3))
        print("  RMSE:", round(result["avg_rmse"], 3))

        if "avg_feature_importance" in result:
            print("  Feature Importances:")
            for idx, importance in enumerate(result["avg_feature_importance"]):
                print(f"    Feature {idx}: {importance:.4f}")

        print()

if __name__ == "__main__":
    main()
    







