import csv
import sys
import os
from collections import defaultdict
from statistics import mean, stdev
from src.config import MAX_WEEKS_BY_YEAR, MATCHUP_FILE
from pprint import pprint

STATISTICS_DIRECTORY = "statistics"

os.makedirs(STATISTICS_DIRECTORY, exist_ok=True)

def load_matchups():
    """
    Loads matchup data from the matchup CSV file.

    Returns:
        list[dict]: List of matchups.
    """
    with open(MATCHUP_FILE, newline="") as f:
        return list(csv.DictReader(f))

def get_streak(results):
    streak = 0
    for result in reversed(results):
        if result == "Tie":
            streak = 0
            break
        elif result == "Win":
            if streak < 0:
                break
            streak += 1
        elif result == "Loss":
            if streak > 0:
                break
            streak -= 1
    return streak

def calculate_stats(matchups, year):  # list of dicts
    weekly_scores = defaultdict(lambda: defaultdict(float))  # team -> week -> score
    results = defaultdict(list)  # team -> list of 'Win', 'Loss', etc.

    for row in matchups:
        if int(row["year"]) != int(year):
            continue
        if row["team_1_result"] == "N/A":  # future weeks
            continue

        week = int(row["week"])
        team_1 = row["team_1"]
        team_2 = row["team_2"]
        try:
            score_1 = float(row["team_1_score"])
            score_2 = float(row["team_2_score"])
        except (ValueError, TypeError):
            continue

        weekly_scores[team_1][week] = score_1
        weekly_scores[team_2][week] = score_2
        results[team_1].append(row["team_1_result"])
        results[team_2].append(row["team_2_result"])

        stats = []
        for team, week_scores in weekly_scores.items():
            scores = [week_scores[w] for w in sorted(week_scores)]
            avg = round(mean(scores), 2)
            std = round(stdev(scores), 2) if len(scores) > 1 else ""
            streak = get_streak(results[team])
            high = max(scores)
            low = min(scores)

            stats.append({
            "Player": team,
            "GP": len(scores),
            "Avg": avg,
            "StDev": std,
            "High": high,
            "Low": low,
            "Streak": streak,
            **{f"Week_{w}": week_scores[w] for w in sorted(week_scores)}
        })

    if not weekly_scores:
        raise ValueError(f"No data in season yet: {year}")
    
    return stats

def result_lookup(matchups, year):  # player -> week -> result (1 for win, 0 for loss)
    result_map = defaultdict(dict)
    for row in matchups:
        if int(row["year"]) != int(year):
            continue
        week = int(row["week"])
        if row["team_1_result"] == "Win":
            result_map[row["team_1"]][week] = 1.0
            result_map[row["team_2"]][week] = 0.0
        elif row["team_1_result"] == "Loss":
            result_map[row["team_1"]][week] = 0.0
            result_map[row["team_2"]][week] = 1.0
        elif row["team_1_result"] == "Tie":
            result_map[row["team_1"]][week] = 0.5
            result_map[row["team_2"]][week] = 0.5
        else:  # result is "N/A":
            continue
    return result_map

def add_luck_index(year, stats, matchups):
    result_map = result_lookup(matchups, year)
    final_week = MAX_WEEKS_BY_YEAR[int(year)]
    sorted_weeks = [f"Week_{i}" for i in range(1, final_week + 1)]
    week_scores = defaultdict(list)

    for row in stats:
        for w in sorted_weeks:
            if w in row and row[w] != "":
                week_scores[w].append(float(row[w]))

    for row in stats:
        total_luck = 0.0
        for w in sorted_weeks:
            if w not in row:
                continue
            player_score = float(row[w])
            scores = week_scores[w]  # scores list for this week w

            num_teams = len(scores) - 1  # total number of other teams for given week w
            num_beaten = sum(1 for s in scores if s < player_score)  # total number of teams given player would beat
            num_tied = sum(1 for s in scores if s == player_score) - 1  # number of teams given player would tie (removes self)
            expected_result = (num_beaten + 0.5 * num_tied) / num_teams if num_teams > 0 else 0
            
            week = int(w.split("_")[1])
            actual_result = result_map[row["Player"]][week]
            total_luck += actual_result - expected_result

        row["LuckIndex"] = round(total_luck, 2)

    return stats

def add_adjusted_avg(year, stats):
    final_week = MAX_WEEKS_BY_YEAR[int(year)]
    sorted_weeks = [f"Week_{i}" for i in range(1, final_week + 1)]

    for row in stats:
        scores = [row[week] for week in sorted_weeks if week in row]
        weights = list(range(1, len(scores) + 1))
        weighted_sum = sum(w * s for w, s in zip(weights, scores))
        weighted_avg = weighted_sum / sum(weights)

        row["AdjustedAvg"] = round(weighted_avg, 2)

    return stats

def add_ewma(year, stats, alpha=0.35):
    final_week = MAX_WEEKS_BY_YEAR[int(year)]
    sorted_weeks = [f"Week_{i}" for i in range(1, final_week + 1)]

    for row in stats:
        scores = [row[week] for week in sorted_weeks if week in row]
        result = scores[0]
        for s in scores[1:]:
            result = alpha * s + (1 - alpha) * result

        row["EWMA"] = round(result, 2)

    return stats
    
def save_stats_csv(year):
    if int(year) not in MAX_WEEKS_BY_YEAR:
        raise ValueError(f"Invalid year: {year}")
    os.makedirs(f"{STATISTICS_DIRECTORY}/{year}", exist_ok=True)
    filename = f"{STATISTICS_DIRECTORY}/{year}/stats.csv"
    matchups = load_matchups()
    try:
        stats = calculate_stats(matchups, year)

        final_week = MAX_WEEKS_BY_YEAR[int(year)]
        sorted_weeks = [f"Week_{i}" for i in range(1, final_week + 1)]

        stats = add_luck_index(year, stats, matchups)
        stats = add_adjusted_avg(year, stats)
        stats = add_ewma(year, stats)
    except ValueError:
        if os.path.isdir(f"{STATISTICS_DIRECTORY}/{year}") and not os.listdir(f"{STATISTICS_DIRECTORY}/{year}"):
            os.rmdir(f"{STATISTICS_DIRECTORY}/{year}")
        raise

    fieldnames = ["Player", "GP", "Avg", "AdjustedAvg", "EWMA", "StDev", "High", "Low", "Streak", "LuckIndex"] + sorted_weeks

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in stats:
            full_row = row.copy()
            for w in sorted_weeks:
                if w not in full_row:
                    full_row[w] = ""
            writer.writerow(full_row)
    print(f"\n{year} stats successfully updated in {filename}.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 -m src.generate_stats <year>")
        sys.exit(1)

    year = sys.argv[1]

    try:
        save_stats_csv(year)
    except ValueError as e:
        print(e)
        sys.exit(1)
