import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


   
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

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

print(out)
print(hidden)


base_dir = "data"

fs = 100
sample = fs * 60

before = 2
after = 2
hr_min = 20
hr_max = 300

names = ["a01"]
y_train = []
x_train = []

labels = wfdb.rdann(os.path.join(base_dir, names[0]), extension="apn").symbol
signals = wfdb.rdrecord(os.path.join(base_dir, names[0]), channels=[0]).p_signal[:, 0]
X = []
y = []
for j in range(len(labels)):
    if j < before or (j + 1 + after) > len(signals) / float(sample):
        continue
    signal = signals[int((j - before) * sample):int((j + 1 + after) * sample)]
    signal, _, _ = st.filter_signal(signal, ftype='FIR', band='bandpass', order=int(0.3 * fs),
                                frequency=[3, 45], sampling_rate=fs)
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
        X.append([(rri_tm, rri_signal), (ampl_tm, ampl_siganl)])
        y.append(0. if labels[j] == 'N' else 1.)


EMBED_DIM = 6
HIDDEN_DIM = 6

model = nn.LSTM(EMBED_DIM, HIDDEN_DIM)
loss_function = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# wfbd in seperate function
for epoch in range(100):
    print("In Epoch")
    for sig, lab in zip(X, y):
        print("in for")

        # clear gradients
        model.zero_grad()
        print("finished zero grad")

        # forward pass 
        ecg_scores = model(torch.tensor(sig))

        print("ecg_scores")
        # loss, gradient, update params 
        loss = loss_function(ecg_scores, lab)
        loss.backward() # only do when there is a label 
        print("loss")
        optimizer.step()
        print("optimizer")

# one minute of ECG data, 1 annotation 
# after each subject, clear memory 
# to get labels:
# labels = wfdb.rdann(os.path.join(base_dir, names[i]), extension="apn").symbol

# to get data:
# signals = wfdb.rdrecord(os.path.join(base_dir, name), channels=[0]).p_signal[:, 0]
