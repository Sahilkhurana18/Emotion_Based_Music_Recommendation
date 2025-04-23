import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Streamlit config
st.set_page_config(page_title="Emotion-Based Music Recommender")
st.title("🎧 Emotion-Based Music Recommender")

# Load model and labels
model = load_model("ResNet50V2_Model.h5", compile=False)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load songs dataset
music_df = pd.read_csv(r"C:\Users\sahil\Downloads\input\music\data_moods.csv")[
    ['name', 'artist', 'mood', 'popularity']
]

# Spotify API credentials (replace with your own)
SPOTIFY_CLIENT_ID = "7a15d998ce1548f38d09d76ac81a95a3"
SPOTIFY_CLIENT_SECRET = "1cfa4ac87f8e4688a78b46c11afdcd8e"

# Authenticate Spotify
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=SPOTIFY_CLIENT_ID,
    client_secret=SPOTIFY_CLIENT_SECRET,
))

# Map emotions to song moods
def Recommend_Songs(emotion):
    mood_map = {
        'Disgust': 'Sad',
        'Happy': 'Happy',
        'Sad': 'Happy',
        'Fear': 'Calm',
        'Angry': 'Calm',
        'Surprise': 'Energetic',
        'Neutral': 'Energetic'
    }
    mood = mood_map.get(emotion, 'Happy')
    filtered = music_df[music_df['mood'] == mood].sort_values(by='popularity', ascending=False).head(20)
    top = filtered.sample(n=5).reset_index(drop=True)
    return top[['name', 'artist']]

# Get Spotify URL
def get_spotify_url(song_name, artist):
    query = f"{song_name} {artist}"
    results = sp.search(q=query, limit=1, type='track')
    items = results.get('tracks', {}).get('items', [])
    if items:
        return items[0]['external_urls']['spotify']
    else:
        return None

# Session state to store emotion
if 'detected_emotion' not in st.session_state:
    st.session_state.detected_emotion = None

FRAME_WINDOW = st.image([])

# Emotion detection button
if st.button('Start Webcam Emotion Detection'):
    cap = cv2.VideoCapture(0)
    last_frame = None

    for _ in range(30):
        ret, frame = cap.read()
        if not ret:
            break
        last_frame = frame

    cap.release()

    if last_frame is not None:
        image = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(image, (224, 224))
        norm = resized.astype("float32") / 255.0
        expanded = np.expand_dims(norm, axis=0)
        preds = model.predict(expanded, verbose=0)
        detected_emotion = emotion_labels[np.argmax(preds)]
        st.session_state.detected_emotion = detected_emotion

        cv2.putText(image, f"{detected_emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        FRAME_WINDOW.image(image)

# Show detected emotion and recommendations
if st.session_state.detected_emotion:
    st.subheader(f"Detected Emotion: {st.session_state.detected_emotion}")

    if st.button("Refresh Recommendations"):
        st.session_state.refresh = True

    if 'refresh' not in st.session_state or st.session_state.refresh:
        st.session_state.refresh = False  # Reset

        st.write("🎵 Recommended Songs:")
        recommendations = Recommend_Songs(st.session_state.detected_emotion)
        for idx, row in recommendations.iterrows():
            url = get_spotify_url(row['name'], row['artist'])
            if url:
                st.markdown(f"{row['name']}** by {row['artist']} — [Listen on Spotify]({url})")
            else:
                st.markdown(f"{row['name']}** by {row['artist']} — Not found on Spotify")