import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

FIGURES_DIRECTORY = "regression/figures"
os.makedirs(FIGURES_DIRECTORY, exist_ok=True)

def model_performance_visualization(results_by_week, save_path=None):
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
    if save_path:
        plt.savefig(f"{save_path}/Model_Performance_r2_rmse.png", bbox_inches="tight", dpi=300)
        plt.close()
    else:
        plt.show()

def model_feature_binary_matrix(SELECTED_FEATURES_BY_WEEK, save_path=None):
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

    if save_path:
        plt.savefig(f"{save_path}/Feature_Binary_Matrix.png", bbox_inches="tight", dpi=300)
        plt.close()
    else:
        plt.show()

def permutation_importance_visualization(perm_importance_by_week, top_n=None, save_path=None):
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
        filename = "Permutation_Importance_All.png"
    else:
        plt.title(f"Top {top_n} Features by Average Permutation Importance")
        filename = f"Permutation_Importance_top{top_n}.png"
    plt.xlabel("Week")
    plt.ylabel("Permutation Importance")
    plt.xticks(df.index)
    plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), fontsize=9)
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        folder = os.path.join(save_path, "Permutation_Importance")
        os.makedirs(folder, exist_ok=True)
        plt.savefig(f"{folder}/{filename}", bbox_inches="tight", dpi=300)
        plt.close()
    else:
        plt.show()

def plot_feature_stability(perm_importance_by_week, threshold=0, save_path=None):
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
    plt.title(f"Feature Stability (Permutation Importance > {round(threshold, 3)})")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    filename = f"Feature_Stability_>{round(threshold, 3)}.png"
    if save_path:
        folder = os.path.join(save_path, "Feature_Stability")
        os.makedirs(folder, exist_ok=True)
        plt.savefig(f"{folder}/{filename}", bbox_inches="tight", dpi=300)
        plt.close()
    else:
        plt.show()

def plot_avg_feature_importances(perm_importance_by_week, top_n=10, save_path=None):
    weeks = sorted(perm_importance_by_week.keys())
    all_features = sorted({f for week in perm_importance_by_week.values() for f in week})

    data = {feature: [perm_importance_by_week[week].get(feature, 0) for week in weeks] for feature in all_features}
    df = pd.DataFrame(data, index=weeks)

    avg_importance = df.mean().sort_values(ascending=False)
    avg_importance = avg_importance.head(top_n).reset_index()
    avg_importance.columns = ["Feature", "AvgImportance"]

    plt.figure(figsize=(10, 6))
    sns.barplot(x="Feature", y="AvgImportance", data=avg_importance, palette="Oranges_r")
    plt.ylabel("Average Permutation Importance")
    plt.title(f"Top {top_n} Features by Average Permutation Importance")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_path:
        plt.savefig(f"{save_path}/Top_10_Avg_Features.png", bbox_inches="tight", dpi=300)
        plt.close()
    else:
        plt.show()