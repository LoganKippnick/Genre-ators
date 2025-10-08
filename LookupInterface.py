import csv

import numpy as np

FILENAME = "second_set.csv"
TRACK_NAME_INDEX = 4
ARTISTS_INDEX = 2

def row_to_song_data(data):
    duration = data[6]
    danceability = data[8]
    energy = data[9]
    key = data[10]
    loudness = data[11]
    mode = data[12]
    speechiness = data[13]
    acousticness = data[14]
    instrumentalness = data[15]
    liveness = data[16]
    tempo = data[18]
    popularity = data[5]
    return np.array(
        [duration, danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness,
         tempo, popularity])

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