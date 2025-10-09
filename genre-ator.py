import math
import pickle

import pandas as pd

import api_interface
import lookup_interface

NUM_TOP_GENRES = 5

model = pickle.load(open("Random Forest.sav", "rb"))

# interface = api_interface
interface = lookup_interface

def predict_genre(song_name, artist):
    data = interface.get_song_data(song_name, artist)
    if data is None:
        print(f"Could not find song {song_name} by {artist}.")
        return

    probs = model.predict_proba(data)[0]
    genres = model.named_steps['clf'].classes_

    results = pd.DataFrame({
        'Genre': genres,
        'Probability': probs
    }).sort_values('Probability', ascending=False).head(NUM_TOP_GENRES)

    print(results.to_string(index=False))

while True:
    song_name = input("Enter song name: ")
    artist = input("Enter artist: ")

    print()
    print(f"=== Prediction for {song_name} ===")
    predict_genre(song_name, artist)
    print()

    again = input("Would you like to try another song? (y/n): ")
    if again.lower() == 'n':
        break