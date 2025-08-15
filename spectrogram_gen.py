import os
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# audio root directory and metadata configuration
audio_root = "/path/to/fma_small"
metadata_file = "/path/to/fma_metadata/tracks.csv"
spectrogram_dir = "spectrograms_out"

# load metadata and filter for 'small' subset
track_meta = pd.read_csv(metadata_file, index_col=0, header=[0, 1])
track_meta = track_meta[track_meta[('set', 'subset')] == 'small']

# select only the genres we are using
genres_to_use = ['Classical', 'Country', 'Hip-Hop', 'Jazz', 'Metal', 'Pop', 'Reggae']
subset_meta = track_meta[track_meta[('track', 'genre_top')].isin(genres_to_use)]

# function to generate and save a mel spectrogram
# **** this function was developed with help from ChatGPT *****
# **** we also referred to librosa: https://librosa.org/doc/main/generated/librosa.feature.melspectrogram.html?utm_source=chatgpt.com ****
def create_and_save_spectrogram(y, sr, save_path, n_mels=128, img_size=277):
    #create a matplotlib figure with no axes
    fig, ax = plt.subplots(figsize=(img_size/100, img_size/100), dpi=100)
    ax.set_axis_off()
    
    # generate mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # display and save the spectrogram
    librosa.display.specshow(mel_db, sr=sr, fmax=8000, ax=ax)
    fig.savefig(save_path, dpi=100, bbox_inches=None, pad_inches=0)
    plt.close(fig)

# loop through each track in the filtered metadata
for tid, row in subset_meta.iterrows():
    genre = row[('track', 'genre_top')]
    genre_dir = os.path.join(spectrogram_dir, genre)
    os.makedirs(genre_dir, exist_ok=True)

    # determine subfolder structure and file path
    tid_str = f"{tid:06d}"
    subfolder = tid_str[:3]
    mp3_file = os.path.join(audio_root, subfolder, f"{tid_str}.mp3")

    # skip if file is missing
    if not os.path.exists(mp3_file):
        print(f"file not found: {mp3_file}")
        continue

    # load audio, generate spectrogram, and save
    try:
        audio, sr = librosa.load(mp3_file, sr=None)
        out_path = os.path.join(genre_dir, f"{genre.lower()}{tid_str}.png")
        create_and_save_spectrogram(audio, sr, out_path)
        print(f"generated: {out_path}")
    except Exception as err:
        print(f"error processing {mp3_file}: {err}")
