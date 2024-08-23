import pybaseball as pyball
from pybaseball import playerid_lookup
from pybaseball import schedule_and_record
from pybaseball import statcast
from pybaseball import statcast_pitcher
from pybaseball import batting_stats
from pybaseball import pitching_stats

# Gather pitcher ID (pitcher_id)
def get_pitcher_id(first_name, last_name):
    """
    first_name: string of pitcher first name
    last_name: string of pitcher last name

    Note:
    key_mlbam: Official MLB player ID
    key_retro: Retro Sheet player ID
    key_bbref: Baseball Reference player ID
    key_fangraphs: Fangraphs player ID
    """
    player_info = playerid_lookup(last_name, first_name)
    key_mlbam = player_info['key_mlbam'].values[0]
    return key_mlbam

# Kershaw Example
kershaw_stats = statcast_pitcher('2017-06-01', '2017-07-01', get_pitcher_id("Clayton", "Kershaw"))
print(kershaw_stats)