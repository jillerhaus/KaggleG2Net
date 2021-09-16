#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np

train_df = pd.read_csv("../input/g2net-gravitational-wave-detection/test_paths.csv")
savedir = "../input/filtered"

import gwpy
from gwpy.timeseries import TimeSeries
from scipy.cluster.vq import whiten
from gwpy.signal import filter_design
from tqdm import tqdm

def save_files(rg_n):
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
        path, ident = train_df.iloc[index][["path", "id"]]
        ts_data = np.load(path).astype(np.float32)

        for i in range(3):
            measurement = ts_data[i]
            ts = TimeSeries(measurement)
            ts.sample_rate=2048
            ts = ts.whiten()

            zpk = filts[i]
            hfilt = ts.filter(zpk, filtfilt=True)
            # hfilt = whiten(hfilt)
            ts_data[i] = np.array(hfilt)

        ts_data = np.transpose(ts_data, [1,0])
        ts_data /= abs(ts_data).max()
        np.save(f"../input/filtered/{ident}.npy", ts_data)

from glob import glob
parts = 16
number = train_df.shape[0]//parts
print(number)
for i in range(parts):
    rg_n = range(i*number,(i+1)*number)
    if glob(f"{savedir}/{train_df.iloc[rg_n[0]].id}.npy") != []:
        print(f"skipping {i}")
    else:
        print(f"doing {i}")
        save_files(rg_n)
