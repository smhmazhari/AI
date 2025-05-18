import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
from glob import glob
import re

import librosa
import librosa.display
import IPython.display as ipd
TOP_DB = 40
audio_files = glob('./voices/recordings/*.wav')
a = audio_files[0]
import re

text = a
pattern = r'(\d+)_([a-zA-Z]+)_(\d+)\.wav'

result = re.search(pattern, text)
groups = result.groups()
dictionary = {
    'number': groups[0],
    'name': groups[1],
    'order': groups[2]
}
def convert_path_to_dict(path):
    pattern = r'(\d+)_([a-zA-Z]+)_(\d+)\.wav'

    result = re.search(pattern, text)
    groups = result.groups()
    return {
        'number': groups[0],
        'name': groups[1],
        'order': groups[2]
    }
def analyse_audio(path):
    output = convert_path_to_dict(path)
    y, sr = librosa.load(path)
    y_tr, index = librosa.effects.trim(y, top_db=TOP_DB)
    mfcc = librosa.feature.mfcc(y=y_tr, sr=sr)
    output['sr'] = sr
    output['y'] = y
    output['y trim'] = y_tr
    output['mfcc'] = mfcc
    return output
# Apply the analyse_audio function to all audio files
output_list = [analyse_audio(file) for file in audio_files[:10]]
# print(output_list)
# Create a DataFrame from the output list
df = pd.DataFrame(output_list)
print(df)