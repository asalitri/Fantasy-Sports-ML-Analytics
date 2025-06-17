import csv
from collections import defaultdict
from statistics import mean, stdev
from src.config import MAX_WEEKS_BY_YEAR

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
        if row["year"] != str(year):
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
    
    return stats

