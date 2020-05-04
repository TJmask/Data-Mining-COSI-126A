import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, Dropout, Activation, BatchNormalization
from tensorflow.keras.utils import to_categorical

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import os
import librosa
import time


def parser(df_input):
    feature = []
    mfcc = []
    label = []
    # Function to load files and extract features
    pre = time.time()
    for i in range(len(df_input)):
        file_name = 'fold' + str(df_input["fold"][i]) + '/' + df_input["slice_file_name"][i]
        # Here kaiser_fast is a technique used for faster extraction
        X, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        # We extract mfcc feature from data
        mels = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)
        mfcc = np.mean(librosa.feature.mfcc(X, sr=sample_rate).T,axis=0)
        mix = np.concatenate([mels, mfcc])
        feature.append(mix)
        label.append(df_input["classID"][i])
        if i % 1000 == 0:
            now = time.time()
            print("==========", i, now - pre)
            pre = now
    return [feature, label]

def transform_data():
    df = pd.read_csv("UrbanSound8K.csv")
    temp = parser(df)
    temp = np.array(temp)
    data = temp.transpose()
    X_ = data[:, 0]
    Y = data[:, 1]

    X = np.empty([len(X_), len(X_[0])])
    for i in range(len(X_)):
        X[i] = (X_[i])
    X = np.array(X)

    data_X = pd.DataFrame(X)
    data_X.to_csv("data_X.csv")
    data_Y = pd.DataFrame(Y)
    data_Y.to_csv("data_Y.csv")

def load_data():
    X = pd.read_csv("data_X.csv")
    X = np.array(X)[:,1:]
    Y = pd.read_csv("data_Y.csv")
    Y = np.array(Y)[:,1]
    return X, Y


def train(X_train, Y_train, X_test, Y_test, k):
    X_train = X_train.reshape(len(X_train), 5, 4, 1)
    X_test = X_test.reshape(len(X_test), 5, 4, 1)
    input_dim = (5, 4, 1)

    dropout_rate = 0.5

    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding = "same", input_shape = input_dim))
    # model.add(BatchNormalization())
    model.add(Activation("tanh"))
    model.add(MaxPool2D())
    model.add(Dropout(dropout_rate))
    model.add(Conv2D(64, (3, 3), padding = "same"))
    # model.add(BatchNormalization())
    model.add(Activation("tanh"))
    model.add(MaxPool2D())
    model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(128))
    # model.add(BatchNormalization())
    model.add(Activation("tanh"))
    model.add(Dropout(dropout_rate))
    model.add(Dense(10))
    model.add(Activation("softmax"))

    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    model.fit(X_train, Y_train, epochs = 50, batch_size = 100, validation_data = (X_test, Y_test))

    print(model.summary())
    predictions = model.predict(X_test)
    score = model.evaluate(X_test, Y_test)
    print(score)


    preds = np.argmax(predictions, axis = 1)
    result = pd.DataFrame(preds)
    result.to_csv("UrbanSound8kResults" + str(k) + ".csv")
    return score[1]


def audio_norm(data):
    max_data = np.max(data)
    min_data = np.min(data)
    data = (data-min_data)/(max_data-min_data+1e-6)
    return data

if __name__ == "__main__":
    X = pd.read_csv("extracted_features_Han_Yue.csv")
    X = np.array(X)
    print(X.shape)
    # if not os.path.exists("data_X.csv"):
    #     transform_data()
    # X, Y = load_data()
    # X = ((X.T-np.mean(X,axis=1))/np.std(X,axis=1)).T
    # Y = to_categorical(Y)

    # folder = pd.read_csv("UrbanSound8K.csv")["fold"]
    # folder = np.array(folder)

    # acc = 0
    # for k in range(1, 2):  # 10-fold
    #     print(k, "=============================================")
    #     idx = np.argwhere(folder!=k).reshape(-1)
    #     X_train = X[idx]
    #     Y_train = Y[idx]
    #     idx = np.argwhere(folder==k).reshape(-1)
    #     X_test = X[idx]
    #     Y_test = Y[idx]

    #     acc += train(X_train[:,128:], Y_train, X_test[:,128:], Y_test, k)
    # acc /= k
    # print("Avg Accuracy:", acc)