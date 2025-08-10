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
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

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

def model_performance_visualization(results_by_week):
    weeks = sorted(results_by_week.keys())
    r2_scores = [results_by_week[w]["avg_r2"] for w in weeks]
    rmse_scores = [results_by_week[w]["avg_rmse"] for w in weeks]

    fig, ax1 = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    ax1[0].plot(weeks, r2_scores, marker='o', color='b')
    ax1[0].set_ylabel("R²", color='b')
    ax1[0].set_title("Model Performance Across Weeks")
    ax1[0].grid(True)

    ax1[1].plot(weeks, rmse_scores, marker='s', color='r')
    ax1[1].set_xlabel("Week")
    ax1[1].set_ylabel("RMSE", color='r')
    ax1[1].grid(True)

    plt.tight_layout()
    plt.show()

def model_feature_binary_matrix():
    all_features = sorted(set(f for features in SELECTED_FEATURES_BY_WEEK.values() for f in features))
    data = []
    for feature in all_features:
        row = [1 if feature in SELECTED_FEATURES_BY_WEEK.get(week, []) else 0 for week in range(3, 10)]
        data.append(row)

    df = pd.DataFrame(data, index=all_features, columns=[f"Week {w}" for w in range(3, 10)])

    df["Total Weeks"] = df.sum(axis=1)
    df = df.sort_values("Total Weeks", ascending=False)

    week_df = df.iloc[:, :-1]
    total_df = df[["Total Weeks"]]

    fig, (ax1, ax2) = plt.subplots(
        ncols=2, 
        sharey=True, 
        gridspec_kw={"width_ratios": [week_df.shape[1], 1]},
        figsize=(8, len(df) * 0.3)
    )

    sns.heatmap(week_df, cmap="Blues", linewidths=0.5, cbar=False, linecolor='gray', annot=True, fmt='d', ax=ax1)
    ax1.set_title("Feature Usage by Week", fontsize=12)
    ax1.set_xlabel("Week")
    ax1.set_ylabel("Feature")

    sns.heatmap(total_df, cmap="Oranges", linewidths=0.5, cbar=False, linecolor='gray', annot=True, fmt='d', ax=ax2)
    ax2.set_title("Total", fontsize=12)
    ax2.set_xlabel("")
    ax2.set_ylabel("")

    plt.tight_layout()
    plt.show()

def feature_stability_summary(perm_importance_by_week, top_n=None):
    weeks = sorted(perm_importance_by_week.keys())
    all_features = sorted({f for week in perm_importance_by_week.values() for f in week})
    data = {feature: [perm_importance_by_week[week].get(feature, 0) for week in weeks] for feature in all_features}
    df = pd.DataFrame(data, index=weeks)

    avg_importance = df.mean().sort_values(ascending=False)

    if top_n is not None:
        df = df[avg_importance.head(top_n).index]

    cmap = plt.get_cmap("tab20", min(len(df.columns), 20))
    line_styles = ["-", "--"]
    markers = ['s', 'o']

    plt.figure(figsize=(14, 7))
    for i, feature in enumerate(df.columns):
        color = cmap(i % 20)
        line_style = line_styles[(i // 20) % len(line_styles)]
        marker = markers[(i // 20) % len(line_styles)]
        plt.plot(df.index, df[feature], marker=marker, linestyle=line_style, label=feature, color=color)

    if top_n is None:
        plt.title("Permutation Importance of All Features Across Weeks")
    else:
        plt.title(f"Top {top_n} Features by Average Permutation Importance")
    plt.xlabel("Week")
    plt.ylabel("Permutation Importance")
    plt.xticks(df.index)
    plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), fontsize=9)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def feature_stability_table_and_plot(perm_importance_by_week, threshold=0.05):
    weeks = sorted(perm_importance_by_week.keys())
    all_features = sorted({f for week in perm_importance_by_week.values() for f in week})

    stability_counts = {}
    for feature in all_features:
        count = sum(1 for week in weeks if perm_importance_by_week[week].get(feature, 0) > threshold)
        stability_counts[feature] = count

    df = pd.DataFrame.from_dict(stability_counts, orient="index", columns=["Weeks Above Threshold"])
    df = df[df["Weeks Above Threshold"] > 0]
    df = df.sort_values(by="Weeks Above Threshold", ascending=False)

    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(len(df))]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(df.index, df["Weeks Above Threshold"], color=colors)
    plt.ylabel("Number of Weeks")
    plt.title(f"Feature Stability (Permutation Importance > {threshold})")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_avg_feature_importances(perm_importance_by_week, top_n=10):
    weeks = sorted(perm_importance_by_week.keys())
    all_features = sorted({f for week in perm_importance_by_week.values() for f in week})

    data = {feature: [perm_importance_by_week[week].get(feature, 0) for week in weeks] for feature in all_features}
    df = pd.DataFrame(data, index=weeks)

    avg_importance = df.mean().sort_values(ascending=False)
    avg_importance = avg_importance.head(top_n)

    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(len(avg_importance))]

    plt.figure(figsize=(10, 6))
    plt.bar(avg_importance.index, avg_importance.values, color=colors)
    plt.ylabel("Average Permutation Importance")
    plt.title(f"Top {top_n} Features by Average Permutation Importance")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def main():
    matchups = load_matchups()

    results_by_week = {}
    perm_importance_by_week = {}
    for week in range(3, 10):
        result = cross_validation_forest_regression(matchups, week, list(LEAGUE_IDS)[:-1], SELECTED_FEATURES_BY_WEEK[week], HYPERPARAMETER_TUNING_BY_WEEK[week])
        results_by_week[week] = result
        perm_importance_by_week[week] = {name: perm_importance for name, perm_importance in zip(result["feature_names"], result["avg_permutation_importance"])}
        
        print(f"Week {week}")
        print("  R²:", round(result["avg_r2"], 3))
        print("  RMSE:", round(result["avg_rmse"], 3))

        if "avg_feature_importance" in result:
            print("  Feature | Permutation Importances:")
            for i, (name, feat_importance, perm_importance) in enumerate(zip(result["feature_names"], result["avg_feature_importance"], result["avg_permutation_importance"])):
                print(f"    {i + 1} {name}: {feat_importance:.4f} | {perm_importance:.4f}")

        print()
        
    model_feature_binary_matrix()

if __name__ == "__main__":
    main()
