#!/usr/bin/env python
# coding: utf-8

# In[5]:


import tensorflow as tf
import pandas as pd
import numpy as np
from glob import glob
import multiprocessing as mp
import os

# In[2]:


INPUT_DIR = "../input/g2net-gravitational-wave-detection"
train_df = pd.read_csv(f"{INPUT_DIR}/training_labels_paths.csv")
test_df = pd.read_csv(f"{INPUT_DIR}/test_paths.csv")
savedir = "../input/whitened-longer-tfrec"


# In[3]:


import gwpy
from gwpy.timeseries import TimeSeries
from scipy.cluster.vq import whiten
from gwpy.signal import filter_design
from tqdm import tqdm
import matplotlib.pyplot as plt


def save_files(rg_n, df=train_df):
    all_data_x = np.zeros([len(rg_n),4096,3], dtype=np.float32)
    all_data_y = np.zeros(len(rg_n), dtype=np.uint8)
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
    print(rg_n)
    slc = df.iloc[rg_n]
    print(slc.head())
    ids = []

    for index in tqdm(range(slc.shape[0])):
        pth, ident, target = slc.iloc[index][["path", "id", "target"]]
        ids.append(ident)
        all_data_y[index] = target
        ts_data = np.load(pth)

        for i in range(3):
            measurement = ts_data[i]
            ts = TimeSeries(measurement, sample_rate=2048)


            zpk = filts[i]
            # ts = ts.filter(zpk, filtfilt=True)
            ts = ts.whiten(1.0, 0.25)
            ts_data[i] = np.array(ts)

        ts_data = np.transpose(ts_data, [1,0])
        ts_data /= abs(ts_data).max()
        # ts_data = ts_data * -1
        ts_data = ts_data.astype(np.float32)
        all_data_x[index] = ts_data


    return all_data_x, all_data_y, ids


# In[8]:




def make_tfrec_train(batches, batch_size, i):

       rg_n = range(i*batch_size,(i+1)*batch_size)
       output_file = f"{savedir}/train{i:02}.tfrec"
       if os.path.exists(output_file):
           print("file already exists")
           return
       print(f"doing {i}")
#         rng = (range(i * batch_size, (i+1)*batch_size))
       tr_x, tr_y, ids = save_files(rg_n, train_df)
       writer = tf.io.TFRecordWriter(output_file)
       for i in tqdm(range(tr_x.shape[0])):
           X = tr_x[i]
           y = tr_y[i]
           ident = str.encode(ids[i])

           # Feature contains a map of string to feature proto objects
           feature = {}
           feature["TimeSeries"] = tf.train.Feature(float_list=tf.train.FloatList(value=X.flatten()))
           feature["id"] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[ident]))
           feature["Target"] = tf.train.Feature(int64_list=tf.train.Int64List(value=[y]))

           #contruct the Example proto object
           example = tf.train.Example(features=tf.train.Features(feature=feature))
           serialized = example.SerializeToString()

           # write the serialized object to disk
           writer.write(serialized)
       writer.close()


# In[10]:


batches = 16
batch_size = train_df.shape[0] // batches
if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    for i in range(batches):
        p = ctx.Process(target = make_tfrec_train, args =(batches, batch_size, i))
        p.start()
        print(f"train {i} started")



test_batches = 8
test_batch_size = test_df.shape[0] // test_batches


def make_tfrec_test(test_batches, test_batch_size, i):
    rg_n = range(i*test_batch_size,(i+1)*test_batch_size)
    print(rg_n)
    output_file = f"{savedir}/test{i:02}.tfrec"
    if os.path.exists(output_file):
        print("test file already exists")
        return
    print(f"doing {i}")
#         rng = (range(i * batch_size, (i+1)*batch_size))
    tr_x, tr_y, ids = save_files(rg_n, test_df)
#         tr_x = test[rg_n]
    writer = tf.io.TFRecordWriter(output_file)
    for i in tqdm(range(tr_x.shape[0])):
        X = tr_x[i]
        ident = str.encode(ids[i])

        # Feature contains a map of string to feature proto objects
        feature = {}
        feature["TimeSeries"] = tf.train.Feature(float_list=tf.train.FloatList(value=X.flatten()))
        feature["id"] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[ident]))

        #contruct the Example proto object
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        serialized = example.SerializeToString()

        # write the serialized object to disk
        writer.write(serialized)
    writer.close()

if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    for i in range(test_batches):
        p = ctx.Process(target = make_tfrec_test, args =(test_batches, test_batch_size, i))
        p.start()
        print(f"test {i} started")
