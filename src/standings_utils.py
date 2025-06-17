def win_pct(wins, ties, gp):
    """
    Calculates win percentage given wins, ties, and games played.

    Args:
        wins (int): Number of wins.
        ties (int): Number of ties.
        gp (int): Games played.

    Returns:
        float: Win percentage.
    """
    return (wins + 0.5 * ties) / gp if gp > 0 else 0.0

def playoff_four_teams(teams_set, data):
    """
    Determines final playoff rankings for a 4-team playoff bracket.

    Args:
        teams_set (set): Set of team names in the playoffs.
        data (list): List of matchup dictionaries.

    Returns:
        dict: Mapping of team names to final ranks (1st to 4th).
    """
    teams_set = teams_set.copy()
    playoff_week_1 = min(int(m["week"]) for m in data)
    week_2_matchups = [m for m in data if int(m["week"]) == playoff_week_1 + 1]
    week_3_matchups = [m for m in data if int(m["week"]) == playoff_week_1 + 2]
    final_rank = {}

    third_place_gm = set()
    for game in week_2_matchups:
        team_1 = game["team_1"]
        team_2 = game["team_2"]
        result = game["team_1_result"]

        if result == "Win":
            loser = team_2
        else:
            loser = team_1

        if loser in teams_set:
            third_place_gm.add(loser)
            teams_set.remove(loser)

    for game in week_3_matchups:
        team_1 = game["team_1"]
        team_2 = game["team_2"]
        result = game["team_1_result"]

        if result == "Win":
            loser = team_2
            winner = team_1
        else:
            loser = team_1
            winner = team_2

        if loser in third_place_gm:
            final_rank[loser] = 4
            final_rank[winner] = 3
        elif loser in teams_set:
            final_rank[loser] = 2
            final_rank[winner] = 1

    return final_rank

def playoff_six_teams(teams_set, data):
    """
    Determines final playoff rankings for a 6-team playoff bracket.

    Args:
        teams_set (set): Set of 6 team names in the playoffs.
        data (list): List of matchup dictionaries.

    Returns:
        dict: Mapping of team names to final ranks (1st to 6th).
    """
    teams_set = teams_set.copy()
    playoff_week_1 = min(int(m["week"]) for m in data)
    week_1_matchups = [m for m in data if int(m["week"]) == playoff_week_1]
    week_2_matchups = [m for m in data if int(m["week"]) == playoff_week_1 + 1]
    week_3_matchups = [m for m in data if int(m["week"]) == playoff_week_1 + 2]
    final_rank = {}

    fifth_place_gm = set()
    for game in week_1_matchups:
        team_1 = game["team_1"]
        team_2 = game["team_2"]
        result = game["team_1_result"]

        if result == "Win":
            loser = team_2
        else:
            loser = team_1

        if loser in teams_set:
            fifth_place_gm.add(loser)
            teams_set.remove(loser)

    championship_gm = set()
    third_place_gm = set()
    for game in week_2_matchups:
        team_1 = game["team_1"]
        team_2 = game["team_2"]
        result = game["team_1_result"]

        if result == "Win":
            loser = team_2
            winner = team_1
        else:
            loser = team_1
            winner = team_2

        if loser in teams_set:
            third_place_gm.add(loser)
            championship_gm.add(winner)
        elif loser in fifth_place_gm:
            final_rank[loser] = 6
            final_rank[winner] = 5

    for game in week_3_matchups:
        team_1 = game["team_1"]
        team_2 = game["team_2"]
        result = game["team_1_result"]

        if result == "Win":
            loser = team_2
            winner = team_1
        else:
            loser = team_1
            winner = team_2

        if loser in championship_gm:
            final_rank[loser] = 2
            final_rank[winner] = 1
        elif loser in third_place_gm:
            final_rank[loser] = 4
            final_rank[winner] = 3
    
    return final_rank

def playoff_seven_teams(teams_set, data):
    """
    Determines final playoff rankings for a 7-team playoff bracket.

    Args:
        teams_set (set): Set of 7 team names in the playoffs.
        data (list): List of matchup dictionaries.

    Returns:
        dict: Mapping of team names to final ranks (1st to 7th).
    """
    teams_set = teams_set.copy()
    playoff_week_1 = min(int(m["week"]) for m in data)
    week_1_matchups = [m for m in data if int(m["week"]) == playoff_week_1]
    week_2_matchups = [m for m in data if int(m["week"]) == playoff_week_1 + 1]
    week_3_matchups = [m for m in data if int(m["week"]) == playoff_week_1 + 2]
    final_rank = {}

    consolation = set()
    for game in week_1_matchups:
        team_1 = game["team_1"]
        team_2 = game["team_2"]
        result = game["team_1_result"]

        if result == "Win":
            loser = team_2
        else:
            loser = team_1

        if loser in teams_set:
            consolation.add(loser)
            teams_set.remove(loser)

    championship_gm = set()
    third_place_gm = set()
    for game in week_2_matchups:
        team_1 = game["team_1"]
        team_2 = game["team_2"]
        result = game["team_1_result"]

        if result == "Win":
            loser = team_2
            winner = team_1
        else:
            loser = team_1
            winner = team_2

        if loser in teams_set:
            third_place_gm.add(loser)
            championship_gm.add(winner)
        elif loser in consolation:
            final_rank[loser] = 7
            consolation.remove(loser)

    for game in week_3_matchups:
        team_1 = game["team_1"]
        team_2 = game["team_2"]
        result = game["team_1_result"]

        if result == "Win":
            loser = team_2
            winner = team_1
        else:
            loser = team_1
            winner = team_2

        if loser in championship_gm:
            final_rank[loser] = 2
            final_rank[winner] = 1
        elif loser in third_place_gm:
            final_rank[loser] = 4
            final_rank[winner] = 3
        elif loser in consolation:
            final_rank[loser] = 6
            final_rank[winner] = 5
    
    return final_rank

def playoff_eight_teams(teams_set, data):
    """
    Determines final playoff rankings for a 8-team playoff bracket.

    Args:
        teams_set (set): Set of 8 team names in the playoffs.
        data (list): List of matchup dictionaries.

    Returns:
        dict: Mapping of team names to final ranks (1st to 8th).
    """
    teams_set = teams_set.copy()
    playoff_week_1 = min(int(m["week"]) for m in data)
    week_1_matchups = [m for m in data if int(m["week"]) == playoff_week_1]
    week_2_matchups = [m for m in data if int(m["week"]) == playoff_week_1 + 1]
    week_3_matchups = [m for m in data if int(m["week"]) == playoff_week_1 + 2]
    final_rank = {}

    consolation = set()
    for game in week_1_matchups:
        team_1 = game["team_1"]
        team_2 = game["team_2"]
        result = game["team_1_result"]

        if result == "Loss":
            loser = team_1
        else:
            loser = team_2

        if loser in teams_set:
            consolation.add(loser)
            teams_set.remove(loser)

    championship_gm = set()
    third_place_gm = set()
    fifth_place_gm = set()
    for game in week_2_matchups:
        team_1 = game["team_1"]
        team_2 = game["team_2"]
        result = game["team_1_result"]

        if result == "Win":
            loser = team_2
            winner = team_1
        else:
            loser = team_1
            winner = team_2

        if loser in teams_set:
            third_place_gm.add(loser)
            championship_gm.add(winner)
        elif loser in consolation:
            fifth_place_gm.add(winner)
            consolation.remove(winner)

    for game in week_3_matchups:
        team_1 = game["team_1"]
        team_2 = game["team_2"]
        result = game["team_1_result"]

        if result == "Win":
            loser = team_2
            winner = team_1
        else:
            loser = team_1
            winner = team_2

        if loser in championship_gm:
            final_rank[loser] = 2
            final_rank[winner] = 1
        elif loser in third_place_gm:
            final_rank[loser] = 4
            final_rank[winner] = 3
        elif loser in fifth_place_gm:
            final_rank[loser] = 6
            final_rank[winner] = 5
        elif loser in consolation:
            final_rank[loser] = 8
            final_rank[winner] = 7
    
    return final_rank