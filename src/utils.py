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

    seen_guids = set()
    for year in range(start_year, end_year + 1):
        league_id = LEAGUE_IDS.get(year)
        if not league_id:
            print(f"Invalid year: {year}")
            return

        league = gm.to_league(league_id)
        settings = league.settings()
        teams = league.teams()

        for team_key, team_info in teams.items():
            team_name = team_info.get("name", "Unnamed Team")
            managers = team_info.get("managers", [])

            for manager_entry in managers:
                manager = manager_entry.get("manager", {})
                nickname = manager.get("nickname", "Unknown")
                guid = manager.get("guid", "Unknown")

                if guid not in seen_guids:
                    print(f"({year}) {team_name} - {nickname}, {guid}")
                    seen_guids.add(guid)

if __name__ == "__main__":
    get_owner_guids()