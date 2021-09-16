#!/usr/bin/env python
# coding: utf-8

# In[16]:


import tensorflow as tf
import pandas as pd
import numpy as np


# In[17]:


INPUT_DIR = "../input/g2net-gravitational-wave-detection"
train_df = pd.read_csv(f"{INPUT_DIR}/training_labels_paths.csv")
test_df = pd.read_csv(f"{INPUT_DIR}/test_paths.csv")
savedir = "../input/filtered_train"


# In[30]:


import gwpy
from gwpy.timeseries import TimeSeries
from scipy.cluster.vq import whiten
from gwpy.signal import filter_design
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

def save_files(rg_n, df=train_df):
    filts=[]
    for j in range(3):
        bp = filter_design.bandpass(50,550, 2048)
        if j == 2:
            notch_freqs = (50, 100, 150)
        else:
            notch_freqs = (60, 120, 180)
        notches = [filter_design.notch(line, 2048) for line in notch_freqs]
        zpk = filter_design.concatenate_zpks(bp, *notches)
        filts.append(zpk)

    for index in tqdm(rg_n):
        path, ident = df.iloc[index][["path", "id"]]
        ts_data = np.load(path).astype(np.float32)

        for i in range(3):
            measurement = ts_data[i]
            ts = TimeSeries(measurement, sample_rate=2048)
            ts = ts.whiten(0.5, 0.25)

            zpk = filts[i]
            hfilt = ts.filter(zpk, filtfilt=True)
            ts_data[i] = np.array(hfilt)

        ts_data = np.transpose(ts_data, [1,0])
        ts_data /= abs(ts_data).max()
        np.save(f"../input/filtered/{ident}.npy", ts_data)



from glob import glob
parts = 16
number = train_df.shape[0]//parts
for i in range(parts):
    rg_n = range(i*number,(i+1)*number)
    if glob(f"{savedir}/{train_df.iloc[rg_n[0]].id}.npy") != []:
        print(f"skipping {i}")
    else:
        print(f"doing {i}")
        save_files(rg_n, train_df)
