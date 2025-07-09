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

###################################################
################change this#######################
DATASET_PATH = "/content/drive/My Drive/aps360/data/"\
GENRES = [ 'jazz', 'pop', 'country']  # update with genres
N_MFCC = 13                 # number of MFCC coefficients to extract
DURATION = 30               # seconds of audio to process
######################################################
######################################################

def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=DURATION)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    return np.mean(mfcc.T, axis=0)  # average over time



X = []
y = []

for genre in GENRES:
    genre_folder = os.path.join(DATASET_PATH, genre)
    for filename in os.listdir(genre_folder):
        if filename.endswith(".wav"):
            filepath = os.path.join(genre_folder, filename)
            try:
                features = extract_features(filepath)
                X.append(features)
                y.append(genre)
            except Exception as e:
                print(f"Error with {filepath}: {e}")

X = np.array(X)
y = np.array(y)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Normalize Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
