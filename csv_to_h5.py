import numpy as np
import pandas as pd
import dask_cudf
import cudf
import cupy as cp
import cuml as ml
import datetime as dt
import h5py
from tqdm import tqdm
import math
import datetime

import subprocess # we will use this to obtain our local IP using the following command
cmd = "hostname --all-ip-addresses"

process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
output, error = process.communicate()
IPADDR = str(output.decode()).split()[0]

from dask_cuda import LocalCUDACluster
cluster = LocalCUDACluster(ip=IPADDR)

from dask.distributed import Client, progress
client = Client(cluster)

batchSize = 90

## bin function for minute
min_step = 5
## bin function for direction
dxn_step = 90

#X and Y image dimensions
lat_res = 495
ln_res = 436

## bin function for latitude
lat_min = 3.840274e+01
lat_max = 3.886540e+01

## bin function for longitude
ln_min = -9.073262e+01
ln_max = -9.018459e+01

##X and Y image resolutions
lat_step = (lat_max - lat_min) / lat_res
ln_step = (ln_max - ln_min) / ln_res

import math


def getNextBatch(batch_index, batch_size, datalist):
    '''Get next slice of data points from a list'''

    data_len = len(datalist)
    start_idx = batch_index * batch_size
    if start_idx < 0: start_idx = 0
    if start_idx > data_len:  start_idx = data_len
    end_idx = start_idx + batch_size
    if end_idx > data_len:  end_idx = data_len

    return datalist[start_idx:end_idx]


def getBatchMaxIter(batch_size, datalist):
    '''Calculate the maximum iterations to loop through to retrieve all batches'''
    return int(math.ceil(len(datalist) / batch_size))


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def printElaspedTime(msg,ed, st):
    t = ed - st
    print("\n[[Time Check]] : Elasped Time {} is: {}\n".format(msg, t))


maxIterBatchSize = getBatchMaxIter(batchSize, codefilters)

t1 = datetime.datetime.now()

for d in s3DayFolderList:
    print("================== Processing Data for DAY {} ==================\n".format(d))

    day_stack_s = np.zeros(495 * 436 * 288 * 4)
    day_stack_v = np.zeros(495 * 436 * 288 * 4)

    t2 = datetime.datetime.now()
    for i in range(maxIterBatchSize):
        print("*********************************************************")
        print("Starting data processing of Day {} - Batch {}".format(d, i))
        print("*********************************************************")

        t3 = datetime.datetime.now()

        # read S3 bucket day folder
        print("...Reading S3 bucket folder for Day {}...".format(d))
        df = readDailyS3CloudData(d)
        # df = df.persist()

        t4 = datetime.datetime.now()
        printElaspedTime("Reading S3 Bucket", t4, t3)

        # get next batch of zip codes to process
        print("...Getting next batch of Zip codes to process...")
        codefilter = getNextBatch(i, batchSize, codefilters)
        print("....Selected Zip codes: ", codefilter)

        t5 = datetime.datetime.now()

        # filter to batch zip codes and create index
        df = df[df['postalCode'].isin(codefilter)]

        t6 = datetime.datetime.now()
        printElaspedTime("Filtering Zip Codes", t6, t5)

        # create unique index for data
        print("...Creating data index...")
        # df['idx'] = df['capturedTimestamp'].astype(str)+ '__' +df['journeyId'].astype(str)
        df['idx'] = df.reset_index().index
        df = df.set_index('idx')

        t7 = datetime.datetime.now()
        printElaspedTime("Creating data index", t7, t6)

        # preproces datetime data
        print("...Preprocessing date and time data...")
        df['day'] = df['capturedTimestamp'].dt.day
        df['year'] = df['capturedTimestamp'].dt.year
        df['month'] = df['capturedTimestamp'].dt.month
        df['hr'] = df['capturedTimestamp'].dt.hour
        df['min'] = df['capturedTimestamp'].dt.minute
        # df['min']  = df['min'].astype(int)

        t8 = datetime.datetime.now()

        # compute minute, heading, latitude, and longitude bins
        print("...Creating time, heading, and spatial bins...")
        df['bin'] = df['min'] // min_step
        df['dxn'] = df['heading'] // dxn_step
        df['lat_bin'] = (df['latitude'] - lat_min) // lat_step
        df['lon_bin'] = (df['longitude'] - ln_min) // ln_step

        t9 = datetime.datetime.now()

        # redefine columns to optimize memory
        print("...Optimizing data structure...")
        df['dxn'] = df['dxn'].astype(np.int16)
        df['lat_bin'] = df['lat_bin'].astype(np.int16)
        df['lon_bin'] = df['lon_bin'].astype(np.int16)
        df['min'] = df['min'].astype(np.int16)
        df['bin'] = df['bin'].astype(np.int16)
        df['journeyId'] = df['journeyId'].astype('category')

        t10 = datetime.datetime.now()
        printElaspedTime("Calculating Min, Direction, Latitude, and Longitude Bins", t10, t7)

        # un-roll data into 1-dimensional index
        print("...Unrolling data into 1D...")
        df['img_bin_index'] = df['lon_bin'] + (ln_res * df['lat_bin'])
        df['idx_stack_vol'] = (495 * 436 * 288 * df['dxn']) + (
                    (((df['hr'] * 12) + df['bin']) * (495 * 436)) + df['img_bin_index'])
        df = df[['idx_stack_vol', 'journeyId', 'speed']]

        t11 = datetime.datetime.now()
        printElaspedTime("Unrolling to 1D data", t11, t10)

        # SPEED
        print("...Calculating SPEED bin index...")
        spdf = df.groupby(["idx_stack_vol"])["speed"].mean().reset_index()
        spdf['speed'] = spdf['speed'].replace(0, 1)
        spdf['speed'] = ((spdf['speed'] - 0) / (spdf['speed'].max())) * 255
        spdf = spdf.compute().to_pandas()
        day_stack_s[spdf['idx_stack_vol']] = spdf['speed']
        del spdf

        t12 = datetime.datetime.now()
        printElaspedTime("Calculating Speed Bin Index", t12, t11)

        # VOLUME
        print("...Calculating VOLUME bin index...")
        df = df[['idx_stack_vol', 'journeyId']]
        df['volume'] = 1
        df = df.drop_duplicates(["idx_stack_vol", "journeyId"])
        df = df.drop(["journeyId"], axis=1)
        df = df.groupby(["idx_stack_vol"]).volume.sum().reset_index()
        df['volume'] = ((df['volume'] - 0) / (df['volume'].max())) * 255
        df['volume'] = ((df['volume'] - 0) / (df['volume'].max())) * 255
        df = df.compute().to_pandas()
        day_stack_v[df['idx_stack_vol']] = df['volume']

        t13 = datetime.datetime.now()
        printElaspedTime("Calculating Volume Bin Index", t13, t12)

        print("*********************************************************")
        print("Completed data processing of Day {} - Batch {}".format(d, i))
        print("*********************************************************")

        # delete dataframe and startover
        del df
        t13a = datetime.datetime.now()
        printElaspedTime("Processing Day {} - Batch {} data".format(d, i), t13a, t3)
        # Show Elasped times

    t14 = datetime.datetime.now()
    printElaspedTime("Processing ALL Day {} data".format(d), t14, t2)

    print("================== Exporting Raw Data for DAY {} ==================".format(d))
    # H5 export
    print("..Preparing Data for HDF5 Export")
    print("..Re-roll data into 3D ...")
    day_stack_s = day_stack_s.reshape((4, 288, 495, 436))
    day_stack_v = day_stack_v.reshape((4, 288, 495, 436))

    array = np.stack((day_stack_v[0], day_stack_s[0], day_stack_v[1], day_stack_s[1], day_stack_v[2], day_stack_s[2],
                      day_stack_v[3], day_stack_s[3]), axis=-1)

    t15 = datetime.datetime.now()
    printElaspedTime("Reshape Data for Export", t15, t14)

    print("...Export 3D data to HDF5...")
    f1 = h5py.File('day{}.h5'.format(d), "w")
    dset1 = f1.create_dataset('dataset_01', dtype='i', data=array)
    f1.close()
    # array.shape

    t16 = datetime.datetime.now()
    printElaspedTime("Export data for day {}".format(d), t16, t15)

    print("================== Smoothing and Exporting Data for DAY {} ==================".format(d))
    fr = h5py.File('day{}.h5'.format(d), 'r+')

    t17 = datetime.datetime.now()
    data = list(fr[list(fr.keys())[0]])
    data1 = data[0:]
    data1 = np.stack(data1, axis=0)

    stack_norm = np.zeros(495 * 436 * 288 * 8)
    stack_norm = stack_norm.reshape((288, 495, 436, 8))

    t18 = datetime.datetime.now()
    for channel in range(0, 7):
        for i in range(0, 494):
            for j in range(0, 435):
                a = data1[:, i, j, channel]
                a = np.insert(moving_average(a), 0, 0)
                a = np.append(a, 0)
                stack_norm[:, i, j, channel] = a

    t19 = datetime.datetime.now()
    printElaspedTime("Smoothing data for day {}".format(d), t19, t16)

    print("..Preparing Smoothed Data for HDF5 Export")
    f2 = h5py.File('day{}_smoothed.h5'.format(d), "w")
    dset2 = f2.create_dataset('dataset_01', dtype='i', data=stack_norm)
    f2.close()

    t20 = datetime.datetime.now()
    printElaspedTime("Export Smooth data for day {}".format(d), t20, t19)

    printElaspedTime("ALL Data ETL and Export data for day {}".format(d), t20, t2)

    print("================== All Data Processing and Export Completed for DAY {} ==================".format(d))
    print("\n\n")

t21 = datetime.datetime.now()
print("================== ALL DATA PROCESSING AND EXPORT COMPLETED ==================")
print("\n\n")
printElaspedTime("ALL Data ETL and Export data for ALL Days", t21, t1)