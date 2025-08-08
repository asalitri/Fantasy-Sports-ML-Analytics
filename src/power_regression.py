import csv
import os
from src.config import MATCHUP_FILE
from src.generate_stats import calculate_stats, add_ewma, add_adjusted_avg, add_luck_index
from src.generate_standings import calculate_standings
from src.power_utils import scaled_metric
from src.config import LEAGUE_IDS, MAX_WEEKS_BY_YEAR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor
import math
import random
import numpy as np
import pandas as pd

SELECTED_FEATURES_BY_WEEK = {
    3: ["stdev", "win_pct", "avg", "pa", "adj_avg_delta", "avg * pa", "avg * win_pct", "adj_avg_delta * pa"],
    4: ["stdev", "streak", "adj_avg", "ewma_avg_delta", "adj_avg_delta", "pf_pa_delta", "pf_pa_delta * adj_avg", "pf_pa_delta * stdev", "pf_pa_delta * streak", "adj_avg * adj_avg_delta"],
    5: ["win_pct", "pa", "high", "range", "range * pf", "pf * win_pct", "luck_index * win_pct"],
    6: ["ewma", "win_pct", "pf", "pa", "high", "low", "luck_index", "range * pf", "range * low", "ewma * luck_index"],
    7: ["ewma", "stdev", "win_pct", "streak", "adj_avg", "pa", "low", "luck_index", "range", "range * luck_index", "low * adj_avg", "range * ewma", "range * low"],
    8: ["ewma", "stdev", "adj_avg", "pf", "ewma_avg_delta", "range", "range * stdev", "range * pf", "stdev * luck_index", "pf * ewma"],
    9: ["avg", "pf", "low", "ewma_avg_delta", "range", "stdev * avg"]
}

HYPERPARAMETER_TUNING_BY_WEEK = {
    3: True,
    4: False,
    5: True,
    6: True,
    7: False,
    8: False,
    9: False
}

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

    rows = []
    for stats_row, standings_row in zip(current_stats, current_standings):
        player = stats_row["Player"]
        try:
            row = {
                "player": player,
                "ewma": float(stats_row["EWMA"]),
                "stdev": float(stats_row["StDev"]),
                "win_pct": float(standings_row[1]["Pct"]),
                "streak": float(stats_row["Streak"]),
                "adj_avg": float(stats_row["AdjustedAvg"]),
                "avg": float(stats_row["Avg"]),
                "pf": float(standings_row[1]["PF"]),
                "pa": float(standings_row[1]["PA"]),
                "high": float(stats_row["High"]),
                "low": float(stats_row["Low"]),
                "luck_index": float(stats_row["LuckIndex"])
            }
            row["target"] = final_wins[player] - current_wins[player]
            rows.append(row)
        except ValueError:
            continue

    df = pd.DataFrame(rows)

    # Derived features
    df["ewma_avg_delta"] = df["ewma"] - df["avg"]
    df["adj_avg_delta"] = df["adj_avg"] - df["avg"]
    df["pf_pa_delta"] = df["pf"] - df["pa"]
    df["range"] = df["high"] - df["low"]
    
    df["avg * pa"] = df["avg"] * df["pa"]  # Week 3
    df["avg * win_pct"] = df["avg"] * df["win_pct"]  # Week 3
    df["adj_avg_delta * pa"] = df["adj_avg_delta"] * df["pa"]  # Week 3

    df["pf_pa_delta * adj_avg"] = df["pf_pa_delta"] * df["adj_avg"]  # Week 4
    df["pf_pa_delta * stdev"] = df["pf_pa_delta"] * df["stdev"]  # Week 4
    df["pf_pa_delta * streak"] = df["pf_pa_delta"] * df["streak"]  # Week 4
    df["adj_avg * adj_avg_delta"] = df["adj_avg"] * df["adj_avg_delta"]  # Week 4

    df["range * pf"] = df["range"] * df["pf"]  # Week 5, Week 6, Week 8
    df["pf * win_pct"] = df["pf"] * df["win_pct"]  # Week 5
    df["luck_index * win_pct"] = df["luck_index"] * df["win_pct"]  # Week 5

    df["range * low"] = df["range"] * df["low"]  # Week 6, Week 7
    df["ewma * luck_index"] = df["ewma"] * df["luck_index"]  # Week 6
    
    df["range * luck_index"] = df["range"] * df["luck_index"]  # Week 7
    df["low * adj_avg"] = df["low"] * df["adj_avg"]  # Week 7
    df["range * ewma"] = df["range"] * df["ewma"]  # Week 7

    df["range * stdev"] = df["range"] * df["stdev"]  # Week 8
    df["stdev * luck_index"] = df["stdev"] * df["luck_index"]  # Week 8
    df["pf * ewma"] = df["pf"] * df["ewma"]  # Week 8

    df["stdev * avg"] = df["stdev"] * df["avg"]  # Week 9

    # Local scaling
    scale_cols = [
        "ewma", "stdev", "adj_avg", "avg", "pf", "pa", "high", "low",
        "ewma_avg_delta", "adj_avg_delta", "pf_pa_delta", "range",
        "avg * pa", "avg * win_pct", "adj_avg_delta * pa",
        "pf_pa_delta * adj_avg", "pf_pa_delta * stdev", "pf_pa_delta * streak", "adj_avg * adj_avg_delta",
        "range * pf", "pf * win_pct", "luck_index * win_pct", "range * low", "ewma * luck_index",
        "range * luck_index", "low * adj_avg", "range * ewma", "range * stdev", "stdev * luck_index",
        "pf * ewma", "stdev * avg"
    ]
    for col in scale_cols:
        df[col] = scaled_metric(df[col])

    return df

def aggregate_features_and_targets(matchups, week, years):
    all_dfs = [extract_features_and_target(matchups, year, week) for year in years]  # list of all dfs
    df = pd.concat(all_dfs, ignore_index=True)  # merges into one df

    # Global scaling
    df["win_pct"] = scaled_metric(df["win_pct"])
    df["streak"] = scaled_metric(df["streak"])
    df["luck_index"] = scaled_metric(df["luck_index"])

    # keep player names and target out, removes bad features
    feature_names = df.drop(columns=["player", "target"]).columns.tolist()

    X = df[feature_names].values
    y = df["target"].values

    return X, y, feature_names

def cross_validation_linear_regression(matchups, week, years):
    r2_scores = []
    rmses = []

    for test_year in years:
        train_years = [y for y in years if y != test_year]

        X_train, y_train, _ = aggregate_features_and_targets(matchups, week, train_years)
        X_test, y_test, _ = aggregate_features_and_targets(matchups, week, [test_year])

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

def get_best_random_forest(X_train, y_train):
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
        "bootstrap": [True, False],
    }

    model = RandomForestRegressor(random_state=42)
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=50,
        cv=5,
        scoring="neg_root_mean_squared_error",
        random_state=42,
        n_jobs=-1,
    )
    
    search.fit(X_train, y_train)
    return search.best_estimator_

def cross_validation_forest_regression(matchups, week, years, selected_features=None, hyperparameter_tuning=False):
    r2_scores = []
    rmses = []
    feature_importances = []
    permutation_importances = []

    _, _, feature_names = aggregate_features_and_targets(matchups, week, [years[0]])

    # If features are specified, gets feature indices
    if selected_features is not None:
        selected_indices = [feature_names.index(f) for f in selected_features]
    else:
        selected_indices = list(range(len(feature_names)))

    for test_year in years:
        train_years = [y for y in years if y != test_year]

        X_train, y_train, _ = aggregate_features_and_targets(matchups, week, train_years)
        X_test, y_test, _ = aggregate_features_and_targets(matchups, week, [test_year])

        # Applies selected features
        X_train = X_train[:, selected_indices]
        X_test = X_test[:, selected_indices]

        if hyperparameter_tuning:
            model = get_best_random_forest(X_train, y_train)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2_scores.append(r2_score(y_test, y_pred))
        rmses.append(np.sqrt(mean_squared_error(y_test, y_pred)))

        feature_importances.append(model.feature_importances_)
        permutation_result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
        permutation_importances.append(permutation_result.importances_mean)

    avg_feature_importance = np.mean(feature_importances, axis=0).tolist()
    avg_permutation_importance = np.mean(permutation_importances, axis=0).tolist()

    return {
        "avg_r2": np.mean(r2_scores),
        "avg_rmse": np.mean(rmses),
        "r2_scores": r2_scores,
        "rmses": rmses,
        "avg_feature_importance": avg_feature_importance,
        "avg_permutation_importance": avg_permutation_importance,
        "feature_names": [feature_names[i] for i in selected_indices]
    }

def xgb_regression(matchups, week, years):
    r2_scores = []
    rmses = []
    feature_importances = []

    _, _, feature_names = aggregate_features_and_targets(matchups, week, [years[0]])

    for test_year in years:
        train_years = [y for y in years if y != test_year]

        X_train, y_train, _ = aggregate_features_and_targets(matchups, week, train_years)
        X_test, y_test , _ = aggregate_features_and_targets(matchups, week, [test_year])

        model = XGBRegressor(
            n_estimators=100,
            learning_rate=0.05,      
            max_depth=3,             
            reg_alpha=1.0,  # L1
            reg_lambda=1.0,  # L2
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
        "avg_feature_importance": avg_feature_importance,
        "feature_names": feature_names
    }

def main():
    matchups = load_matchups()

    for week in range(3, 10):
        result = cross_validation_forest_regression(matchups, week, list(LEAGUE_IDS)[:-1], SELECTED_FEATURES_BY_WEEK[week], HYPERPARAMETER_TUNING_BY_WEEK[week])

        print(f"Week {week}")
        print("  R²:", round(result["avg_r2"], 3))
        print("  RMSE:", round(result["avg_rmse"], 3))

        if "avg_feature_importance" in result:
            print("  Feature | Permutation Importances:")
            for i, (name, feat_importance, perm_importance) in enumerate(zip(result["feature_names"], result["avg_feature_importance"], result["avg_permutation_importance"])):
                print(f"    {i + 1} {name}: {feat_importance:.4f} | {perm_importance:.4f}")

        print()

if __name__ == "__main__":
    main()
