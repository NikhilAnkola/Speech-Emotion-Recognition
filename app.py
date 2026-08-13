import json
from pathlib import Path

import numpy as np
import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf

st.set_page_config(page_title="Speech Emotion Recognition", page_icon="🎙️", layout="centered")

MODEL_DIR = Path(__file__).parent / "model"


@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(MODEL_DIR / "hybrid_cnn_bilstm_final_v2.keras")
    with open(MODEL_DIR / "label_map.json") as f:
        label_map = json.load(f)
    with open(MODEL_DIR / "feature_config.json") as f:
        feature_config = json.load(f)
    emotions = [label_map[str(i)] for i in range(len(label_map))]
    return model, emotions, feature_config


model, EMOTIONS, cfg = load_artifacts()

SR = cfg["sr"]
DURATION = cfg["duration"]
N_SAMPLES = int(SR * DURATION)
N_MELS = cfg["n_mels"]
N_FFT = cfg["n_fft"]
HOP_LENGTH = cfg["hop_length"]
N_MFCC = cfg["n_mfcc"]


# --- Same preprocessing / feature extraction as the training notebook ---

def normalize_audio(audio):
    peak = np.max(np.abs(audio))
    return audio / peak if peak > 0 else audio


def fix_length(audio, n_samples=N_SAMPLES):
    if len(audio) < n_samples:
        audio = np.pad(audio, (0, n_samples - len(audio)))
    else:
        audio = audio[:n_samples]
    return audio


def extract_features(audio, sr=SR):
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)

    return log_mel.astype(np.float32), mfcc.astype(np.float32)


def predict_emotion(audio_file):
    audio, _ = librosa.load(audio_file, sr=SR, mono=True)
    audio = normalize_audio(audio)
    audio = fix_length(audio)

    mel, mfcc = extract_features(audio)
    mel_in = mel[np.newaxis, ..., np.newaxis]
    mfcc_in = mfcc.T[np.newaxis, ...]

    probs = model.predict([mel_in, mfcc_in], verbose=0)[0]
    return probs, mel


# --- UI ---

st.title("🎙️ Speech Emotion Recognition")
st.caption("Hybrid CNN-BiLSTM model trained on RAVDESS")

uploaded_file = st.file_uploader("Upload a WAV audio clip", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file)

    with st.spinner("Analyzing..."):
        probs, mel = predict_emotion(uploaded_file)

    pred_idx = int(np.argmax(probs))
    pred_emotion = EMOTIONS[pred_idx]
    confidence = probs[pred_idx]

    st.subheader(f"Predicted emotion: **{pred_emotion.capitalize()}**")
    st.write(f"Confidence: {confidence:.1%}")

    sorted_pairs = sorted(zip(EMOTIONS, probs), key=lambda x: -x[1])
    labels = [p[0].capitalize() for p in sorted_pairs]
    values = [p[1] for p in sorted_pairs]

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(labels[::-1], values[::-1])
    ax.set_xlabel("Confidence")
    ax.set_xlim(0, 1)
    st.pyplot(fig)

    with st.expander("View log-mel spectrogram"):
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        librosa.display.specshow(mel, x_axis="time", ax=ax2)
        st.pyplot(fig2)
else:
    st.info("Upload a .wav file to get started.")
