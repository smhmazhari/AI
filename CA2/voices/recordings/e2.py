
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
audio_files = glob('./voices/recordings/*.wav')
# ipd.Audio(audio_file)

genre_lists = {}
genre_lists_trans = {}
# # segment_duration = 10
for file in range(0,60):
    mfcc_list = []
    mfcc_trans = []
    for j in range(0,50):
        signal, sr = librosa.load(audio_files[file])
        mfccs = librosa.feature.mfcc(y=signal, n_mfcc=13, sr=sr)
        mfcc_list.append(mfccs)
        mfcc_transposed = mfccs.T 
        mfcc_trans.append(mfcc_transposed)
        name = audio_files[file*50 + j]
        under  = name.rindex("_")
        name = name[22:under]
        # print(name)

    genre_lists[name] = mfcc_list
    genre_lists_trans [name] = mfcc_trans
    # if (j == 1 ):
    #     plt.figure(figsize=(25, 10))
    #     plt.title(f"MFCC Heatmap - {name}")
    #     librosa.display.specshow(mfccs, 
    #                             x_axis="time",
    #                             sr=sr)
    #     plt.colorbar(format="%+2.f")
    #     # plt.show()
    #     # print(name)
    #     plt.savefig(f"{name}")#f"mffcs_heatmap/{name}.png"
    #     plt.close()
# # print (genre_lists["theo"])
# # print(genre_lists)
# # print(genre_lists_trans)
# print("hello")
# print(len(genre_lists))
# print(genre_lists_trans.keys())
import numpy as np
from hmmlearn import hmm
from sklearn.model_selection import train_test_split
import pickle
import math
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import matplotlib.patches as mpatches
#my new code
import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import LabelEncoder
import librosa

global hmm_models
hmm_models = {}

def make_hmm_model(genre , genre_lists_trans):
    X = np.array([])
    Y = []
    train, test = train_test_split(genre_lists_trans[genre] , test_size=0.2, shuffle = False)
    hmm_model = hmm.GaussianHMM(n_components = 6,covariance_type = 'diag' ,n_iter = 10)
    # print((train))
    # print((test))
    lenghts = []
    for i in train:
        if(len(X)==0):
            X = i
        else:
            X = np.append(X,i,axis=0)
        lenghts.append(len(i))

        # print(i)
    np.seterr(all='ignore')
    print("shape x:",X.shape)
    hmm_model.fit(X,lengths=lenghts)
    hmm_models[genre] = hmm_model
    return hmm_model


        

def predict(x_test):
    pred_labels = []
    for i in range(len(x_test)):
        max_score = -math.inf
        max_label = None
        for genre in hmm_models:
            hmm_model = hmm_models[genre]
            score = hmm_model.score(x_test[i])
            print (i , genre , score)
            if(score > max_score):
                max_score = score
                max_label = genre
        pred_labels.append(max_label)
    return pred_labels


for genre in genre_lists_trans.keys():
    if(len(genre_lists_trans[genre]) > 0):
        make_hmm_model(genre , genre_lists_trans)

for genre in genre_lists.keys():
    # pred_lables = []
    train, test = train_test_split(genre_lists_trans[genre] , test_size=0.2, shuffle = False)
    print(predict(test))
def make_confusion_matrix(genre_lists):
    classes = ["george", "jackson", "lucas", "nicolas","theo","yweweler"]
    pred_labels = []
    real_labels = []
    for genre in genre_lists.keys():
        if len(genre_lists[genre]) > 0:
            train, test = train_test_split(genre_lists_trans[genre], test_size=0.2, shuffle=False)
            pred_labels += predict(test)
            real_labels += [genre] * len(test)
    cm = confusion_matrix(real_labels, pred_labels)
    # Create a heatmap
    sns.set(font_scale=1.4) 
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()
    return cm

cm = make_confusion_matrix(genre_lists_trans )

col_sum = [cm[0][i] + cm[1][i] + cm[2][i] + cm[3][i]+cm[4][i]+cm[5][i] for i in range(6)]
precision = [cm[i][i]/sum(cm[i]) for i in range(0,6)]
recalls = [cm[i][i] / col_sum[i] for i in range(6)]
F1_score = [(2*precision[i]*recalls[i])/ (precision[i] + recalls[i]) for i in range(6)]
accuracy = sum(cm[i][i] for i in range(6))/sum(sum(cm))
plt.clf()
genres = list(genre_lists_trans.keys())#[1:]
print(len(precision))
print(len(genres))
plt.scatter(genres, precision)
plt.scatter(genres, recalls)
plt.scatter(genres, F1_score)
plt.grid()
plt.xlabel("genres")
plt.ylabel("metric")
blue_patch = mpatches.Patch(color='blue',label='precision')
orange_patch = mpatches.Patch(color='orange',label='F1_score')
green_patch = mpatches.Patch(color='green',label='recalls')
plt.legend(handles=[blue_patch,orange_patch,green_patch])

# Adding the exact values on the plot
for i, genre in enumerate(genres):
    plt.text(genre, precision[i], f'{precision[i]:.2f}', ha='center', va='bottom')
    plt.text(genre, recalls[i], f'{recalls[i]:.2f}', ha='center', va='bottom')
    plt.text(genre, F1_score[i], f'{F1_score[i]:.2f}', ha='center', va='bottom')

plt.show()


for i, genre in enumerate(genres):
    print(f"{genre}:")
    print(f"Precision: {precision[i]:.2f}")
    print(f"Recall: {recalls[i]:.2f}")
    print(f"F1 Score: {F1_score[i]:.2f}")
    print()
    
# weighted_average = sum(F1_score)/4 
# micro_average = (sum(cm[i][i] for i in range(6)) / sum(sum(cm))) + (sum(cm[i][i] for i in range(6)) / sum(sum(cm))) / 2 
# macro_average = sum(F1_score)/4   
# print("micro average: " + str(micro_average))
# print("weighted_average: " + str(weighted_average))
# print("macro average: " + str(macro_average))
print("accuract: " + str(accuracy))