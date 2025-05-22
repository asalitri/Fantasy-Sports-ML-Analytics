from yahoo_oauth import OAuth2
from yahoo_fantasy_api import Game

def get_league_ids(start_year=2017, end_year=2026):
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

if __name__ == "__main__":
    get_league_ids()