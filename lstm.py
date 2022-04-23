from builtins import breakpoint
from re import L
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


   
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch.nn.functional as F
import biosppy.signals.tools as st
import numpy as np
import os
import wfdb
from biosppy.signals.ecg import correct_rpeaks, hamilton_segmenter
from scipy.signal import medfilt
# from sklearn.utils import cpu_count
from tqdm import tqdm

torch.manual_seed(1)

lstm = nn.LSTM(900, 2)
inputs = [torch.randn(1, 900) for x in range(5)]

hidden = (torch.randn(1, 1, 2),
         torch.randn(1, 1, 2))

for i in inputs:
    out, hidden = lstm(i.view(1, 1, -1), hidden)

# print(out)
# print(hidden)


base_dir = "data"

fs = 100
sample = fs * 60

before = 2
after = 2
hr_min = 20
hr_max = 300

names_train = ["a01"]
names_test = ["x01"]




labels = wfdb.rdann(os.path.join(base_dir, names_train[0]), extension="apn").symbol
signals = wfdb.rdrecord(os.path.join(base_dir, names_train[0]), channels=[0]).p_signal[:, 0]


labels_test = wfdb.rdann(os.path.join(base_dir, names_test[0]), extension="apn").symbol
signals_test = wfdb.rdrecord(os.path.join(base_dir, names_test[0]), channels=[0]).p_signal[:, 0]


def preprocess(labels, signals):
    X = []
    y = []
    print("LABELS", len(labels))
    for j in range(len(labels)):
        # print("THIS IS J", j)
        if j < before or (j + 1 + after) > len(signals) / float(sample):
            continue
        # print(len(signals))
        signal = signals[int((j - before) * sample):int((j + 1 + after) * sample)]
        # print(len(signal))
        signal, _, _ = st.filter_signal(signal, ftype='FIR', band='bandpass', order=int(0.3 * fs),
                                        frequency=[3, 45], sampling_rate=fs)
        # Find R peaks
        rpeaks, = hamilton_segmenter(signal, sampling_rate=fs)
        rpeaks, = correct_rpeaks(signal, rpeaks=rpeaks, sampling_rate=fs, tol=0.1)
        if len(rpeaks) / (1 + after + before) < 40 or \
                len(rpeaks) / (1 + after + before) > 200:  # Remove abnormal R peaks signal
            continue
        # Extract RRI, Ampl signal
        rri_tm, rri_signal = rpeaks[1:] / float(fs), np.diff(rpeaks) / float(fs)
        rri_signal = medfilt(rri_signal, kernel_size=3)
        ampl_tm, ampl_siganl = rpeaks / float(fs), signal[rpeaks]
        hr = 60 / rri_signal
        # Remove physiologically impossible HR signal
        if np.all(np.logical_and(hr >= hr_min, hr <= hr_max)):
            # Save extracted signal
            appending = list(rri_tm) + list(rri_signal) + list(ampl_tm) + list(ampl_siganl)
            appending = appending[:1470]
            if len(appending) < 1470:
                zeros = 1470 - len(appending)
                zeroslist = [0] * zeros
                appending += zeroslist
            X.append(appending)
            y.append(0. if labels[j] == 'N' else 1.)
        # X.append([signal])
        # y.append(0. if labels[j] == 'N' else 1.)
    return X, y

X, y = preprocess(labels, signals)
X_test, y_test = preprocess(labels_test, signals_test)



EMBED_DIM = 1
HIDDEN_DIM = 2

# in forward, call lstm
# nn.linear (hidden_dim, 1)
# sigmoid
# out = out[-1] in forward + unsqueeze

class LSTMECG(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(LSTMECG, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden = nn.Linear(hidden_dim, 1)
    def forward(self, sig):
        # print("HII",sig.shape)
        lstm_out, _ = self.lstm(sig.view(sig.shape[0], sig.shape[-1], -1))
        lstm_out = lstm_out[:, -1, :] #.unsqueeze(0)
        # print(lstm_out.shape)
        new = self.hidden(lstm_out)
        score = torch.sigmoid(new)
        return score

model = LSTMECG(EMBED_DIM, HIDDEN_DIM)
loss_function = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
# import pdb; pdb.set_trace()
# print([len(X[i]) for i in range(len(X))])
# X = np.array(X)
# print(X.shape)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
accu = []

# print("X", X)
# print("y", y)
# print(X.shape, y.shape)
# wfbd in seperate function
for epoch in range(50):
    print("In Epoch", epoch)
    for sig, lab in zip(X, y):
        # print("in for")

        # clear gradients
        model.zero_grad()
        # print("finished zero grad")

        # forward pass 
        sigT = sig.t()
        
        ecg_scores = model(sig.unsqueeze(0))
        # ecg_scores = ecg_scores.item()
        lab = lab.unsqueeze(0).unsqueeze(0) #torch.tensor(lab)

        # print("ecg_scores")
        # loss, gradient, update params 
        # print("ecg_scores", ecg_scores)
        # print(lab)
        loss = loss_function(ecg_scores, lab)
        loss.backward() # only do when there is a label 
        # print("loss")
        optimizer.step()
        # print("optimizer")

    if epoch % 5 == 0:
        with torch.no_grad():
            acc = 0
            total = 0
            for sig, lab in zip(X_test, y_test):
                total += 1
                # sigT = sig.t()
                score = model(sig.unsqueeze(0))
                if score > 0.5: score = 1
                else: score = 0
                print(score, lab)
                if score == lab:
                    acc += 1
            print(acc/total)
            accu.append(acc/total)

# torch save dict

a = open("accuracy.txt", "w")
for i in accu:
    a.write(str(i))
    a.write("\n")

torch.save(model.state_dict(), "model1")
