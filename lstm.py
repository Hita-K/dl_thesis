import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

lstm = nn.LSTM(900, 2)
inputs = [torch.randn(1, 900) for x in range(5)]

hidden = (torch.randn(1, 1, 2),
         torch.randn(1, 1, 2))

for i in inputs:
    out, hidden = lstm(i.view(1, 1, -1), hidden)

print(out)
print(hidden)


   
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import biosppy.signals.tools as st
import numpy as np
import os
import wfdb
from biosppy.signals.ecg import correct_rpeaks, hamilton_segmenter
from scipy.signal import medfilt
from sklearn.utils import cpu_count
from tqdm import tqdm

def preprocess(name, label):
    idxs = [label[e] for e in name]
    return torch.tensor(idxs, dtype=torch.long)

def read_text_file(file_path):
    with open(file_path, 'r') as f:
        print(f.read())

training_data = [read_text_file(f) for f in os.listdir()]

apnea_ecg = {}
for el, tag in training_data:
    for w in el:
        if w not in apnea_ecg:
            apnea_ecg[w] = len(apnea_ecg) #unique id

tags = { "a01": 1, "a02": 2, "a03": 3, "a04": 4, "a05": 5}
EMBED_DIM = 6
HIDDEN_DIM = 6

class LSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, data_size, out_size):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.ecg_embeddings = nn.Embedding(data_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, out_size)

    def forward(self, ecg):
        embeds = self.ecg_embeddings(ecg)
        lstm_out, _ = self.lstm(embeds.view(len(ecg), 1, -1))
        ecg_space = self.hidden2tag(lstm_out.view(len(ecg), -1))
        ecg_scores = F.log_softmax(ecg_space, dim=1)
        return ecg_scores

    
model = LSTM(EMBED_DIM, HIDDEN_DIM, len(apnea_ecg), len(tags))
loss_function = nn.BCELoss() #check this

with torch.no_grad():
    inputs = preprocess(training_data[0][0], apnea_ecg)
    ecg_score = model(inputs)
    print(ecg_score)
base_dir = "dataset"
names = [
    "a01", "a02", "a03", "a04", "a05", "a06", "a07", "a08", "a09", "a10",
    "a11", "a12", "a13", "a14", "a15", "a16", "a17", "a18", "a19", "a20",
    "b01", "b02", "b03", "b04", "b05",
    "c01", "c02", "c03", "c04", "c05", "c06", "c07", "c08", "c09", "c10"
]
fs = 100
sample = fs * 60
before = 2
after = 2
# wfbd in seperate function
for epoch in range(100):
    for i, el in enumerate(training_data):
        signals = wfdb.rdrecord(os.path.join(base_dir, el), channels=[0]).p_signal[:, 0]

        labels = wfdb.rdann(os.path.join(base_dir, names[i]), extension="apn").symbol
        for j in tqdm(range(len(labels)), desc=el, file=sys.stdout):
            if j < before or \
                    (j + 1 + after) > len(signals) / float(sample):
                continue
            signal = signals[int((j - before) * sample):int((j + 1 + after) * sample)]



        # clear gradients
        model.zero_grad()

        # convert inputs into tensors
        ecg_input = preprocess(signals)
        targets = preprocess(labels)

        # forward pass 
        ecg_scores1 = model(ecg_input)

        # loss, gradient, update params 
        loss = loss_function(ecg_scores1, targets)
        loss.backward() # only do when there is a label 
        optimizer.step()

# one minute of ECG data, 1 annotation 
# after each subject, clear memory 
# to get labels:
# labels = wfdb.rdann(os.path.join(base_dir, names[i]), extension="apn").symbol

# to get data:
# signals = wfdb.rdrecord(os.path.join(base_dir, name), channels=[0]).p_signal[:, 0]
