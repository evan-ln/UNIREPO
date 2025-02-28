import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment

# Define input files (MP3)
mp3_songs = ["evan-l-1.mp3", "evan-l-2.mp3"]

# Convert MP3 to WAV and process
for mp3_song in mp3_songs:
    # Convert MP3 to WAV
    wav_song = mp3_song.replace(".mp3", ".wav")
    audio = AudioSegment.from_mp3(mp3_song)
    audio.export(wav_song, format="wav")

    # Load WAV file
    y, sr = librosa.load(wav_song)

    # Compute chromagram
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    # Plot and save the chromagram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(chroma, x_axis="time", y_axis="chroma", cmap="coolwarm")
    plt.colorbar()
    plt.title(f"Chromagram - {wav_song}")
    plt.savefig(f"{wav_song}_chromagram.png", dpi=300)
    plt.close()

print("Chromagrams saved successfully.")

