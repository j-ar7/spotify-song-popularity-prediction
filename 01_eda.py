import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import re

def get_prim_art(id_str):
	id_list = eval(id_str)
	if id_list:
		return id_list[0]
	else:
		return None

def genre_dict_sort(gen_str):
	if gen_str not in genres:
		genres.update({gen_str:1})
	else:
		genres[gen_str] += 1

def get_prim_genre(genre):
	if type(genre) == float:
		return "unknown"
	else:
		genre = eval(genre)
		if genre: return genre[0]
		else: return "unknown"

def get_year(dt):
	pattern1 = re.compile(r'^(\d{4})-\d{2}-\d{2}$')
	match1 = pattern1.match(dt)
	if match1:
		return match1.group(1)
	else:
		pattern2 = re.compile(r'^(\d{4})-\d{2}$')
		match2 = pattern2.match(dt)
		if match2:
			return match2.group(1)
		else:
			pattern3 = re.compile(r'^(\d{4})$')
			match3 = pattern3.match(dt)
			if match3:
				return match3.group(1)
			else: print(dt)

artists_df = pd.read_csv('./data/artists.csv')
tracks_df = pd.read_csv('./data/tracks.csv')

tracks_df = tracks_df.dropna(subset=["name"]) # removed all entries where name = NaN; manually checked that there are 71 entries with no name and no artists and they all have the same artist id. checked the id on spotify, likely pollutant entries.

# nan_in_tracks_df = tracks_df[tracks_df.isnull().any(axis=1)]
# print(nan_in_tracks_df.to_string())

tracks_df['primary_artist_id'] = tracks_df['id_artists'].apply(get_prim_art)
tracks_df['primary_artist'] = tracks_df['artists'].apply(get_prim_art)
tracks_df = tracks_df.drop(['id_artists', 'artists'], axis=1)
tracks_df.rename(columns={'id':'track_id', 'name':'track_name', 'popularity':'track_popularity'}, inplace=True)

# prim_verify = tracks_df[(tracks_df['id_artists'].str.len() > 26) | (tracks_df['artists'].str.len() > 26)] # manually got the figure 26
# print(prim_verify['id', 'name', 'artists', 'id_artists', 'primary_id_artists', 'primary_artists'].head(80))

main_df = pd.merge(tracks_df, artists_df, left_on="primary_artist_id", right_on="id", how="left", suffixes=('_track', '_artist'))
main_df = main_df.drop(['id', 'name'], axis=1)
main_df.rename(columns={'popularity':'artist_popularity'}, inplace=True)

# -----------
# genre reports
# genres = {}
# main_df['genres'].apply(genre_dict_sort)

# plt.figure(dpi=1200)
# plt.plot(list(genres.keys()), list(genres.values()))
# plt.savefig('genre_dist.png')

# genres = {key:value for key, value in sorted(genres.items(), key=lambda item: item[1])}

# plt.figure(dpi=1200)
# plt.plot(list(genres.keys()), list(genres.values()))
# plt.savefig('genre_dist_sorted.png')
# -----------

main_df['primary_genre'] = main_df['genres'].apply(get_prim_genre)
main_df = main_df.drop(['genres'], axis=1)

# -------------
# follower reports
# foll_df = main_df[main_df['followers'].notnull()]
# # foll_dict = dict(zip(foll_df['primary_artists'], foll_df['followers']))

# # plt.figure(dpi=1200)
# # plt.plot([i.replace("$", "_") for i in list(foll_dict.keys())], list(foll_dict.values()))
# # plt.savefig('foll_dist.png')

# foll = sorted(list(foll_df['followers']))
# plt.plot(range(len(foll)), foll)
# plt.savefig('foll_curve.png')

# p = int(12.5/100 * len(foll))
# foll = foll[p-1:-p]
# truncated_mean = int(sum(foll)/len(foll))
# gpt says median is still a better measure for central tendency so i will be going with that instead
#------------

median_foll = main_df['followers'].median()
main_df['followers'].fillna(median_foll, inplace=True)

main_df['release_year'] = main_df['release_date'].apply(get_year)
main_df = main_df.drop(['release_date'], axis=1)

main_df = main_df.dropna(subset=['artist_popularity']) # since nan entries in artist_popularity is only 2% of the total size, im gonna drop it instead of imputing

model_features = [
'track_popularity','duration_ms', 'explicit', 'danceability', 'energy',
'key', 'loudness', 'mode', 'speechiness', 'acousticness',
'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature',
'followers', 'artist_popularity', 'primary_genre', 'release_year']
#excluded features: track_id, track_name, primary_artist_id, primary_artist

model_df = main_df[model_features].copy()

os.makedirs('./processed_data', exist_ok=True)
output_path = './processed_data/model_data.csv'
model_df.to_csv(output_path, index=False)