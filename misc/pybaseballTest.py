from pybaseball import (
    statcast, statcast_pitcher, statcast_batter,
    playerid_lookup, playerid_reverse_lookup,
    batting_stats, pitching_stats, team_batting, team_pitching,
    standings, schedule_and_record, top_prospects
)
import pandas as pd


def describe_df(name: str, df: pd.DataFrame):
    print(f"\n=== {name} ===")
    print(f"Number of rows: {len(df)}")
    print("Columns and dtypes:")
    print(df.dtypes)
    print("-" * 40)


def main():
    # Reverse lookup
    player_df = playerid_reverse_lookup([545361])
    describe_df("Player ID Reverse Lookup", player_df)

    # Statcast: full pitch-by-pitch data
    try:
        statcast_df = statcast("2024-04-01", "2024-04-03")
        describe_df("Statcast (Pitch-by-pitch)", statcast_df)
    except Exception as e:
        print("Error loading statcast data:", e)

    # Statcast pitcher-specific
    try:
        sc_pitcher_df = statcast_pitcher("2024-04-01", "2024-04-03", 545361)
        describe_df("Statcast Pitcher View", sc_pitcher_df)
    except Exception as e:
        print("Error loading statcast pitcher data:", e)

    # Statcast batter-specific
    try:
        sc_batter_df = statcast_batter("2024-04-01", "2024-04-03", 545361)
        describe_df("Statcast Batter View", sc_batter_df)
    except Exception as e:
        print("Error loading statcast batter data:", e)

    # Batting Stats
    try:
        batting_df = batting_stats(2023)
        describe_df("Season Batting Stats", batting_df)
    except Exception as e:
        print("Error loading batting stats:", e)

    # Pitching Stats
    try:
        pitching_df = pitching_stats(2023)
        describe_df("Season Pitching Stats", pitching_df)
    except Exception as e:
        print("Error loading pitching stats:", e)

    # Team Batting
    try:
        team_batting_df = team_batting(2023)
        describe_df("Team Batting Stats", team_batting_df)
    except Exception as e:
        print("Error loading team batting stats:", e)

    # Team Pitching
    try:
        team_pitching_df = team_pitching(2023)
        describe_df("Team Pitching Stats", team_pitching_df)
    except Exception as e:
        print("Error loading team pitching stats:", e)

    # Standings
    try:
        standings_df = standings(2023)
        describe_df("MLB Standings", standings_df)
    except Exception as e:
        print("Error loading standings:", e)

    # Schedule & Record
    try:
        schedule_df = schedule_and_record(2023, "HOU")
        describe_df("Team Schedule & Record", schedule_df)
    except Exception as e:
        print("Error loading schedule & record:", e)

    # Top Prospects
    try:
        prospects_df = top_prospects()
        describe_df("Top MLB Prospects", prospects_df)
    except Exception as e:
        print("Error loading top prospects:", e)


if __name__ == "__main__":
    main()
