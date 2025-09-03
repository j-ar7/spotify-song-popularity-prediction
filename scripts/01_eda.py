import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import re
from sklearn.preprocessing import MultiLabelBinarizer

def get_year(dt):
	pattern1 = re.compile(r'^(\d{4})-\d{2}-\d{2}$')
	match1 = pattern1.match(dt)
	if match1:
		return int(match1.group(1))
	else:
		pattern2 = re.compile(r'^(\d{4})-\d{2}$')
		match2 = pattern2.match(dt)
		if match2:
			return int(match2.group(1))
		else:
			pattern3 = re.compile(r'^(\d{4})$')
			match3 = pattern3.match(dt)
			if match3:
				return int(match3.group(1))
			else: print(dt)

def combpopu(id_str):
	id_list = eval(id_str)
	if not isinstance(id_list, list) or not id_list:
		return None

	valid_popularities = [
		artist_popu_map[artist_id]
		for artist_id in id_list
		if artist_id in artist_popu_map
    ]

	return sum(valid_popularities) if valid_popularities else None

def combfoll(id_str):
	id_list = eval(id_str)
	if not isinstance(id_list, list) or not id_list:
		return None

	valid_followings = [
		artist_foll_map[artist_id]
		for artist_id in id_list
		if artist_id in artist_foll_map and not np.isnan(artist_foll_map[artist_id])
    ]

	return sum(valid_followings) if valid_followings else None

def parse_list_string(s):
    try:
        return eval(s)
    except (TypeError, SyntaxError):
        return []

def get_all_genres(id_str):
	id_list = eval(id_str)
	if not id_list:
		return []

	all_genres = []
	for artist_id in id_list:
		genres = artist_genre_map.get(artist_id, [])
		all_genres.extend(genres)

	return list(set(all_genres))

def genre_lis(id_list):
	for genre in id_list:
		if genre in all_genres_ls:
			all_genres_ls[genre] += 1
		else:
			all_genres_ls.update({genre:1})

def filter_top_genres(genre_list):
    return [genre for genre in genre_list if genre in top_genres]


artists_df = pd.read_csv('./data/artists.csv')
tracks_df = pd.read_csv('./data/tracks.csv')

# # nan_in_tracks_df = tracks_df[tracks_df.isnull().any(axis=1)]
# # print(nan_in_tracks_df.to_string())

artists_df['genres'] = artists_df['genres'].apply(parse_list_string)
artist_genre_map = artists_df.set_index('id')['genres'].to_dict()

artist_popu_map = artists_df.set_index('id')['popularity'].to_dict()
artist_foll_map = artists_df.set_index('id')['followers'].to_dict()

tracks_df['combined_artist_popularity'] = tracks_df['id_artists'].apply(combpopu)
tracks_df['combined_following'] = tracks_df['id_artists'].apply(combfoll)
tracks_df['combined_genres'] = tracks_df['id_artists'].apply(get_all_genres)
tracks_df['release_year'] = tracks_df['release_date'].apply(get_year)

all_genres_ls = {}
tracks_df['combined_genres'].apply(genre_lis)

TOP_N_GENRES = 800
sorted_genres = sorted(all_genres_ls.items(), key=lambda item: item[1], reverse=True)
top_genres = [genre for genre, count in sorted_genres[:TOP_N_GENRES]]

tracks_df['filtered_genres'] = tracks_df['combined_genres'].apply(filter_top_genres)

mlb = MultiLabelBinarizer(classes=top_genres)
genre_encoded_df = pd.DataFrame(
	mlb.fit_transform(tracks_df.pop('filtered_genres')),
	columns=mlb.classes_,
	index=tracks_df.index
)

main_df = pd.concat([tracks_df, genre_encoded_df], axis=1)

main_df = main_df.dropna(subset=['combined_artist_popularity']) # nan entries < 2%, no need to impute; same for combined_following as well
main_df.drop(['combined_genres', 'id', 'name', 'artists', 'id_artists', 'release_date'], axis=1, inplace=True) # excluded features

main_df.info()
os.makedirs('./processed_data', exist_ok=True)
output_path = './processed_data/model_data.csv'
main_df.to_csv(output_path, index=False)