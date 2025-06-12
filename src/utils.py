import sys
from yahoo_oauth import OAuth2
from yahoo_fantasy_api import Game
from src.config import LEAGUE_IDS

def get_league_ids(start_year=2017, end_year=2025):
    oauth = OAuth2(None, None, from_file='oauth2.json')
    gm = Game(oauth, 'nfl')

    for year in range(start_year, end_year + 1):
        try:
            league_ids = gm.league_ids(year)
            if league_ids:
                for league_id in league_ids:
                    league = gm.to_league(league_id)
                    settings = league.settings()
                    name = settings.get('name')
                    print(f"{year} - {name}: {league_id}")
            else:
                print(f"{year}: No leagues found.")
        except Exception as e:
            print(f"{year}: Error — {e}")

def get_owner_guids(start_year=2017, end_year=2025):
    oauth = OAuth2(None, None, from_file='oauth2.json')
    gm = Game(oauth, 'nfl')

    for index, year in enumerate(range(start_year, end_year + 1)):
        league_id = LEAGUE_IDS.get(year)
        if not league_id:
            print(f"Invalid year: {year}")
            return

        league = gm.to_league(league_id)
        settings = league.settings()
        teams = league.teams()

        team_counter_yearly = 0
        for team_key, team_info in teams.items():
            team_name = team_info.get("name", "Unnamed Team")
            managers = team_info.get("managers", [])

            for manager_entry in managers:
                team_counter_yearly += 1
                manager = manager_entry.get("manager", {})
                nickname = manager.get("nickname", "Unknown")
                guid = manager.get("guid", "Unknown")

                print(f"({year}) {team_counter_yearly}. {team_name} - {nickname}, {guid}")
        if index != end_year - start_year:
            print("------------------------------")  # Separates each league year

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage:")
        print("  python -m src.utils league_ids")
        print("  python -m src.utils owner_guids")
        sys.exit(1)

    command = sys.argv[1]

    if command == "league_ids":
        get_league_ids()
    elif command == "owner_guids":
        get_owner_guids()
    else:
        print("Unknown command. Use 'league_ids' or 'owner_guids'.")
        sys.exit(1)