import pybaseball as pyball
from pybaseball import playerid_lookup
from pybaseball import schedule_and_record
from pybaseball import statcast
from pybaseball import statcast_pitcher
from pybaseball import batting_stats
from pybaseball import pitching_stats

# Gather pitcher ID (pitcher_id)
# dfPitcher = playerid_lookup('Darvish', 'Yu')
# print("pitcher_id: ")
# print(dfPitcher)
# """
# key_mlbam: Official MLB player ID
# key_retro: Retro Sheet player ID
# key_bbref: Baseball Reference player ID
# key_fangraphs: Fangraphs player ID
# """

# Gather Game ID (game_pk)
# game_data = statcast(start_dt='2023-04-01', end_dt='2023-04-01')
# if game_data['home_team'].isin(['LAD']).any():
#     row = game_data.query('home_team == "LAD"') # Check if 'LAD' is the home or away team
# else:
#     row = game_data.query('away_team == "LAD"') # Get the unique game_pk for the specific game
# game_pk = row['game_pk'].unique()
# print("game_pk:")
# print(game_pk)

# Gather pitcher arsenal (pitch_type)
# pitcher_id = 506433  # Replace with the actual pitcher's ID
# start_date = '2023-01-01'
# end_date = '2023-12-31'
# pitch_data = statcast_pitcher(start_dt = start_date, end_dt = end_date, player_id = pitcher_id)
# pitch_arsenal = pitch_data['pitch_type'].unique()
# print("pitch_type: ")
# print(pitch_arsenal)



# Gather specific game date (date)
# team_schedule = schedule_and_record(2023, 'LAD')
# specific_game = team_schedule[team_schedule['Date'] == '2023-04-01']  # Replace with the actual date
# print("game date: ")
# print(specific_game['Date'])