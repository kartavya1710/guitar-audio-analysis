import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
from pydub import AudioSegment
import pyloudnorm as pyln
import io
import noisereduce as nr
from scipy import signal
from scipy.spatial.distance import cdist

# Function to load and preprocess audio (cleaning and amplification)
def preprocess_audio(file):
    y, sr = librosa.load(file, sr=None)
    
    # Noise reduction
    y_clean = nr.reduce_noise(y=y, sr=sr)
    
    # Normalize loudness
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(y_clean)
    y_normalized = pyln.normalize.loudness(y_clean, loudness, -23.0)
    
    return y_normalized, sr

# Function to calculate chroma features
def get_chroma(y, sr):
    hop_length = 512
    n_fft = 2048
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    chroma = librosa.feature.chroma_stft(S=S, sr=sr, n_fft=n_fft, hop_length=hop_length)
    return np.nan_to_num(chroma)  # Replace NaN with 0

# Function to calculate similarity using DTW on chroma features
# DTW is similarity algorithm used

def calculate_similarity(y1, y2, sr):
    chroma1 = get_chroma(y1, sr)
    chroma2 = get_chroma(y2, sr)
    
    # Ensure non-zero values in chroma features
    epsilon = 1e-6
    chroma1 = np.maximum(chroma1, epsilon)
    chroma2 = np.maximum(chroma2, epsilon)
    
    # Calculate cosine distance between chroma features
    dist_matrix = cdist(chroma1.T, chroma2.T, metric='cosine')
    
    # Perform DTW on the distance matrix
    D, wp = librosa.sequence.dtw(C=dist_matrix, step_sizes_sigma=np.array([[1, 1], [0, 1], [1, 0]]))
    
    # Calculate similarity based on the DTW path
    path_length = len(wp)
    normalized_distance = np.sum([dist_matrix[i, j] for i, j in wp]) / path_length
    similarity = (1 - normalized_distance) * 100
    
    return similarity

# Function to find delay points
def find_delay_points(y1, y2, sr):
    chroma1 = get_chroma(y1, sr)
    chroma2 = get_chroma(y2, sr)
    
    correlation = np.correlate(np.mean(chroma1, axis=0), np.mean(chroma2, axis=0), mode='same')
    delay_points = np.where(correlation < np.percentile(correlation, 10))[0]
    return librosa.frames_to_time(delay_points, sr=sr, hop_length=512)

# Function to plot waveform with enhanced visuals
def plot_waveform(y, sr, title, delay_points=None):
    fig, ax = plt.subplots(figsize=(12, 4))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title(title)
    if delay_points is not None:
        for delay in delay_points:
            ax.axvline(x=delay, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    plt.tight_layout()
    return fig

# Streamlit app setup
st.title("Enhanced Guitar Audio Comparison Tool")
st.markdown("Compare your guitar playing with the original track using advanced audio analysis.")

# File upload
user_audio = st.file_uploader("Upload your guitar audio", type=["mp3", "wav"])
original_audio = st.file_uploader("Upload the original audio", type=["mp3", "wav"])

if user_audio and original_audio:
    # Preprocess both audios
    user_y, user_sr = preprocess_audio(user_audio)
    original_y, original_sr = preprocess_audio(original_audio)

    # Ensure both audios are the same sample rate and length
    if user_sr != original_sr:
        original_y = librosa.resample(original_y, original_sr, user_sr)
    min_length = min(len(user_y), len(original_y))
    user_y = user_y[:min_length]
    original_y = original_y[:min_length]

    # Calculate similarity
    similarity = calculate_similarity(user_y, original_y, user_sr)

    # Find delay points
    delay_points = find_delay_points(user_y, original_y, user_sr)

    # Display results
    st.subheader(f"Similarity: {similarity:.2f}%")
    
    # Plot waveforms
    st.pyplot(plot_waveform(user_y, user_sr, "Your Guitar Audio"))
    st.pyplot(plot_waveform(original_y, user_sr, "Original Audio"))

    # Plot waveform with delays
    st.subheader("Delays in Hitting Chords")
    st.pyplot(plot_waveform(user_y, user_sr, "Your Guitar Audio with Detected Delays", delay_points))

    # Display chroma features
    st.subheader("Chroma Features Comparison")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    librosa.display.specshow(get_chroma(user_y, user_sr), y_axis='chroma', x_axis='time', ax=ax1)
    ax1.set_title("Your Guitar Audio Chroma")
    librosa.display.specshow(get_chroma(original_y, user_sr), y_axis='chroma', x_axis='time', ax=ax2)
    ax2.set_title("Original Audio Chroma")
    plt.tight_layout()
    st.pyplot(fig)

else:
    st.warning("Please upload both your guitar audio and the original audio to proceed.")