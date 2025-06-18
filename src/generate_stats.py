import csv
import sys
import os
from collections import defaultdict
from statistics import mean, stdev
from src.config import MAX_WEEKS_BY_YEAR, MATCHUP_FILE

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

def get_win_streak(results):
    streak = 0
    for result in reversed(results):
        if result == "Win":
            streak += 1
        else:
            break
    return streak

def calculate_stats(matchups, year):
    weekly_scores = defaultdict(lambda: defaultdict(float))  # team -> week -> score
    results = defaultdict(list)  # team -> list of 'Win', 'Loss', etc.

    for row in matchups:
        if int(row["year"]) != int(year):
            continue
        if row["team_1_result"] == "N/A":
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
            avg = mean(scores)
            std = stdev(scores) if len(scores) > 1 else 0.0
            win_streak = get_win_streak(results[team])

            stats.append({
            "Player": team,
            "GP": len(scores),
            "Avg": avg,
            "StDev": std,
            "WinStreak": win_streak,
            **{f"Week_{w}": week_scores[w] for w in sorted(week_scores)}
        })

    if not weekly_scores:
        raise ValueError(f"No data in season yet: {year}")
    
    return stats

def save_stats_csv(year):
    if int(year) not in MAX_WEEKS_BY_YEAR:
        raise ValueError(f"Invalid year: {year}")
    os.makedirs(f"{STATISTICS_DIRECTORY}/{year}", exist_ok=True)
    filename = f"{STATISTICS_DIRECTORY}/{year}/stats.csv"
    matchups = load_matchups()
    try:
        stats = calculate_stats(matchups, year)
    except ValueError:
        if os.path.isdir(f"{STATISTICS_DIRECTORY}/{year}") and not os.listdir(f"{STATISTICS_DIRECTORY}/{year}"):
            os.rmdir(f"{STATISTICS_DIRECTORY}/{year}")
        raise

    final_week = MAX_WEEKS_BY_YEAR[int(year)]
    sorted_weeks = [f"Week_{i}" for i in range(1, final_week + 1)]
    fieldnames = ["Player", "GP", "Avg", "StDev", "WinStreak"] + sorted_weeks

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
