import pandas as pd

def load_scores(filepath):
    return pd.read_csv(filepath)

def calculate_standings(df):
    standings = []
    for team in df['team'].unique():
        team_data = df[df['team'] == team]
        wins = (team_data['points'] > team_data['opponent_points']).sum()
        losses = (team_data['points'] < team_data['opponent_points']).sum()
        standings.append({'team': team, 'wins': wins, 'losses': losses})
    return pd.DataFrame(standings).sort_values(by='wins', ascending=False)

if __name__ == "__main__":
    df = load_scores("data/weekly_scores_template.csv")
    standings_df = calculate_standings(df)
    print(standings_df)
