
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
audio_files = glob('./voices/recordings/*.wav')
# ipd.Audio(audio_file)

# genre_lists = {}
# genre_lists_trans = {}
# mfcc_list = []
# mfcc_trans = []
# # segment_duration = 10
# for file in range(0,len(audio_files)):

#     signal, sr = librosa.load(audio_files[file])
#     mfccs = librosa.feature.mfcc(y=signal, n_mfcc=13, sr=sr)
#     mfcc_list.append(mfccs)
#     mfcc_transposed = mfccs.T 
#     mfcc_trans.append(mfcc_transposed)
#     name = audio_files[file]
#     # under = name.rindex("_")
#     name = name[20:-4]
#     # name = name[20:under]
#     # print(name)
#     if name not in genre_lists :
#         genre_lists.update({name:mfcc_list})
#         genre_lists_trans.update({name:mfcc_trans})
#     else:
#         genre_lists[name] += mfcc_list
#         genre_lists_trans[name] += mfcc_trans
#     # if (file % 50 == 1 ):
#     #     plt.figure(figsize=(25, 10))
#     #     plt.title(f"MFCC Heatmap - {name}")
#     #     librosa.display.specshow(mfccs, 
#     #                             x_axis="time",
#     #                             sr=sr)
#     #     plt.colorbar(format="%+2.f")
#     #     # plt.show()
#     #     # print(name)
#     #     plt.savefig(f"{name}")#f"mffcs_heatmap/{name}.png"
#     #     plt.close()
# # print (genre_lists["theo"])
# # print(genre_lists)
# # print(genre_lists_trans)
# print("hello")

#my new code
import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import LabelEncoder
import librosa

# Function to extract features from audio file
def extract_features(audio_path):
    y, sr = librosa.load(audio_path)
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    return mfccs.T
numbers = ['zero', 'one','two','three','four','five','six','seven','eight','nine']  # Add more labels as needed
# Prepare your dataset
# X - list of feature arrays, y - list of corresponding labels (spoken numbers)
X = []
y = []
for file in range(0,len(audio_files)):
    # if (file % 50 < 39):
        X.append(extract_features(audio_files[file]))
        y.append(numbers[file // 300])

# Encode labels to integers
le = LabelEncoder()
y_encoded = le.fit_transform(numbers)
# print(y_encoded)
# Initialize HMM parameters
n_components = 5  # Number of states in the HMM
# n_mix = 3  # Number of mixtures for each state
covariance_type = 'diag'  # Type of covariance parameters
n_iter = 1000  # Number of iterations to run the algorithm

# Initialize Gaussian HMM
model = hmm.GaussianHMM(n_components=n_components, covariance_type=covariance_type, n_iter=n_iter)

# Train HMM on features (you may need to concatenate your features and lengths for real data)
model.fit(np.concatenate(X), lengths=[len(x) for x in X])
   
# Predict the spoken number in a new audio file
# new_audio_path = 'path/to/new_audio.wav'
# new_features = extract_features(new_audio_path)
logprob, sequence = model.decode(extract_features(audio_files[2999]))
predicted_label = le.inverse_transform([sequence[0]])

print(f"The spoken number is: {predicted_label[0]}")



























# import numpy as np
# from hmmlearn import hmm
# from sklearn.model_selection import train_test_split
# import pickle
# import math
# from sklearn.metrics import confusion_matrix
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler
# import matplotlib.patches as mpatches

# global hmm_models
# hmm_models = {}
# def make_hmm_model(genre , genre_lists_trans):
#     X = np.array([])
#     Y = []
#     train, test = train_test_split(genre_lists_trans[genre] , test_size=0.2, shuffle = False)
#     hmm_model = hmm.GaussianHMM(n_components = 10,covariance_type = 'diag' ,n_iter = 10)
#     for i in train:
#         if(len(X)==0):
#             X = i
#         else:
#             X = np.append(X,i,axis=0)
#     np.seterr(all='ignore')
#     hmm_model.fit(X)
#     hmm_models[genre] = hmm_model
#     # with open("hmm_models/" + str(genre) + '_hmm_model.pkl', 'wb') as f:
#     #     pickle.dump(hmm_models[genre], f)
#     return hmm_model

# for genre in genre_lists_trans.keys():
#     if(len(genre_lists_trans[genre]) > 0):
#         make_hmm_model(genre , genre_lists_trans)

# def predict(x_test):
#     pred_labels = []
#     for i in range(len(x_test)):
#         max_score = -math.inf
#         max_label = None
#         for genre in hmm_models:
#             hmm_model = hmm_models[genre]
#             score = hmm_model.score(x_test[i])
#             print (i , genre , score)
#             if(score > max_score):
#                 max_score = score
#                 max_label = genre
#         pred_labels.append(max_label)
#     return pred_labels

# def make_confusion_matrix(genre_lists):
#     classes = ["george", "jackson", "lucas", "nicolas","theo","yweweler"]
#     pred_labels = []
#     real_labels = []
#     for genre in genre_lists.keys():
#         if len(genre_lists[genre]) > 0:
#             train, test = train_test_split(genre_lists[genre], test_size=0.2, shuffle=False)
#             pred_labels += predict(test)
#             real_labels += [genre] * len(test)
#     cm = confusion_matrix(real_labels, pred_labels)
#     # Create a heatmap
#     sns.set(font_scale=1.4) 
#     plt.figure(figsize=(10, 7))
#     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=classes, yticklabels=classes)
#     plt.xlabel("Predicted Labels")
#     plt.ylabel("True Labels")
#     plt.title("Confusion Matrix")
#     plt.show()
#     return cm

# cm = make_confusion_matrix(genre_lists_trans )

