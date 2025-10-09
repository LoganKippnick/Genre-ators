import csv

import numpy as np
import pandas as pd

FILENAME = "second_set.csv"
TRACK_NAME_INDEX = 4
ARTISTS_INDEX = 2

def row_to_song_data(data):
    return pd.DataFrame([{
        'Duration': data[6],
        'Danceability': data[8],
        'Energy': data[9],
        'Key': data[10],
        'Loudness': data[11],
        'Mode': data[12],
        'Speechiness': data[13],
        'Acousticness': data[14],
        'Instrumentalness': data[15],
        'Liveness': data[16],
        'Tempo': data[18],
        'Popularity': data[5]
    }])

def get_song_data(song_name, artist):
    with open(FILENAME, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[TRACK_NAME_INDEX] == song_name:
                artists = row[ARTISTS_INDEX].split(';')
                for a in artists:
                    if a == artist:
                        return row_to_song_data(row)
        return None