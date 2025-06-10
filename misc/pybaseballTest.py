from pybaseball import playerid_reverse_lookup

player_info_df = playerid_reverse_lookup([545361])
name = player_info_df.iloc[0]['name_first'] + " " + player_info_df.iloc[0]['name_last']
print(name)
