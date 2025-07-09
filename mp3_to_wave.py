import os
import numpy as np
import librosa
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import subprocess


from google.colab import drive
drive.mount('/content/drive')



##################change this########################
folder = "/content/drive/My Drive/aps360/data/"

for root, dirs, files in os.walk(folder):
    for filename in files:
        if filename.endswith(".mp3"):
            mp3_path = os.path.join(root, filename)
            wav_filename = filename.replace(".mp3", ".wav")
            wav_path = os.path.join(root, wav_filename)

            # Convert using ffmpeg
            result = subprocess.run(
                ["ffmpeg", "-y", "-i", mp3_path, wav_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

            if os.path.exists(wav_path):
                os.remove(mp3_path)
                print(f"✅ Converted and removed: {mp3_path}")
            else:
                print(f"❌ Conversion failed: {mp3_path}")
