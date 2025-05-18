import numpy as np
from hmmlearn import hmm
import librosa
from glob import glob
# Assuming you have training data for each digit (0-9)
audio_files = glob('./voices/recordings/*.wav')
# Create an HMM model for each digit
def extract_features(audio_path):
    y, sr = librosa.load(audio_path)
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    return mfccs.T

models = []
for digit in range(10):
    model = hmm.GaussianHMM(n_components=3, covariance_type="diag")
    for j in range(300):
        X = extract_features(audio_files[digit *  300 + j])
        model.fit(X)
    # Fit the model with your training data for this digit
    # Replace 'X' with your actual training data
    models.append(model)

def recognize_digit(audio_features):
    # Given audio features, recognize the digit
    likelihoods = [model.score(audio_features) for model in models]
    recognized_digit = np.argmax(likelihoods)
    return recognized_digit

# Example usage:
for speaker in range(1, 7):
    for digit in range(10):
        audio_features = extract_features(f"speaker{speaker}_digit{digit}.wav")
        recognized_digit = recognize_digit(audio_features)
        print(f"Speaker {speaker}, Digit {digit}: Recognized as {recognized_digit}")
