from unittest import case

import numpy as np
import pandas as pd
import spotipy
import requests
import base64
from spotipy.oauth2 import SpotifyClientCredentials

CLIENT_ID = 'fa2d1a0ffef24017a66d336a7c08763a'
CLIENT_SECRET = '0a33a0c9f426479ea7eab9cafbbf4283'

RAPIDAPI_HEADERS = {
# 	"x-rapidapi-key": "424efaf7a0msh4492745b3fdd84cp1a4ef2jsn8679d82dc7b7",
# 	"x-rapidapi-key": "6ccf2eba09msh8fe0a7cee920773p1287f7jsn7f4e0b9b1573",
	"x-rapidapi-key": "243e197429msh44f2c170b1ed304p13adfajsnc893a4288b2a",
	"x-rapidapi-host": "track-analysis.p.rapidapi.com"
}

client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

def get_access_token(client_id, client_secret):
    auth_string = f"{client_id}:{client_secret}"
    auth_bytes = auth_string.encode("utf-8")
    auth_base64 = base64.b64encode(auth_bytes).decode("utf-8")

    token_url = "https://accounts.spotify.com/api/token"
    headers = {
        "Authorization": f"Basic {auth_base64}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {"grant_type": "client_credentials"}

    response = requests.post(token_url, headers=headers, data=data)
    response.raise_for_status()
    return response.json().get("access_token")

def get_track_id(song_name, artist, access_token):
    search_url = "https://api.spotify.com/v1/search"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {
        "q": f"track:\"{song_name}\" artist:\"{artist}\"",
        "type": "track",
        "limit": 1
    }

    response = requests.get(search_url, headers=headers, params=params)
    response.raise_for_status()
    data = response.json()

    if data and data['tracks']['items']:
        return data['tracks']['items'][0]['id']
    return None

def duration_to_ms(length):
    [minutes, seconds] = length.split(":")
    return (int(minutes) * 60 + int(seconds)) * 1000

def note_to_num(note):
    match note:
        case 'C':
            return 0
        case 'C#':
            return 1
        case 'D':
            return 2
        case 'D#':
            return 3
        case 'E':
            return 4
        case 'F':
            return 5
        case 'F#':
            return 6
        case 'G':
            return 7
        case 'G#':
            return 8
        case 'A':
            return 9
        case 'A#':
            return 10
        case 'B':
            return 11
        case _:
            return None

def mode_to_num(mode):
    match mode:
        case 'minor':
            return 0
        case 'major':
            return 1
        case _:
            return None

def song_data_array(data_json):
    return pd.DataFrame([{
        'Duration': duration_to_ms(data_json['duration']),
        'Danceability': data_json['danceability'] * 0.01,
        'Energy': data_json['energy'] * 0.01,
        'Key': note_to_num(data_json['key']),
        'Loudness': int(data_json['loudness'][:-3]),
        'Mode': mode_to_num(data_json['mode']),
        'Speechiness': data_json['speechiness'] * 0.01,
        'Acousticness': data_json['acousticness'] * 0.01,
        'Instrumentalness': data_json['instrumentalness'] * 0.01,
        'Liveness': data_json['liveness'] * 0.01,
        'Tempo': data_json['tempo'],
        'Popularity': data_json['popularity']
    }])

    return np.array([duration, danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, tempo, popularity])

def get_song_data(song_name, artist):
    track_id = get_track_id(song_name, artist, get_access_token(CLIENT_ID, CLIENT_SECRET))
    return song_data_array(requests.get(f'https://track-analysis.p.rapidapi.com/pktx/spotify/{track_id}/', headers=RAPIDAPI_HEADERS).json())
