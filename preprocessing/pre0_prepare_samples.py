import warnings
warnings.simplefilter('ignore')

import os
import re
import gc
import glob
import time

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from sklearn.model_selection import train_test_split

x_files = [
    './data_xy/x_0001_02.npy',
    './data_xy/x_0001_03.npy',
    './data_xy/x_0001_04.npy',
    './data_xy/x_0001_05.npy',
    './data_xy/x_0001_06.npy',
    './data_xy/x_0001_07.npy',
    './data_xy/x_0001_08.npy',
    './data_xy/x_0001_09.npy',
    './data_xy/x_0001_10.npy',
    './data_xy/x_0001_11.npy',
    './data_xy/x_0001_12.npy',
    './data_xy/x_0002_01.npy',
    './data_xy/x_0002_02.npy',
    './data_xy/x_0002_03.npy',
    './data_xy/x_0002_04.npy',
    './data_xy/x_0002_05.npy',
    './data_xy/x_0002_06.npy',
    './data_xy/x_0002_07.npy',
    './data_xy/x_0002_08.npy',
    './data_xy/x_0002_09.npy',
    './data_xy/x_0002_10.npy',
    './data_xy/x_0002_11.npy',
    './data_xy/x_0002_12.npy',
    './data_xy/x_0003_01.npy',
    './data_xy/x_0003_02.npy',
    './data_xy/x_0003_03.npy',
    './data_xy/x_0003_04.npy',
    './data_xy/x_0003_05.npy',
    './data_xy/x_0003_06.npy',
    './data_xy/x_0003_07.npy',
    './data_xy/x_0003_08.npy',
    './data_xy/x_0003_09.npy',
    './data_xy/x_0003_10.npy',
    './data_xy/x_0003_11.npy',
    './data_xy/x_0003_12.npy',
    './data_xy/x_0004_01.npy',
    './data_xy/x_0004_02.npy',
    './data_xy/x_0004_03.npy',
    './data_xy/x_0004_04.npy',
    './data_xy/x_0004_05.npy',
    './data_xy/x_0004_06.npy',
    './data_xy/x_0004_07.npy',
    './data_xy/x_0004_08.npy',
    './data_xy/x_0004_09.npy',
    './data_xy/x_0004_10.npy',
    './data_xy/x_0004_11.npy',
    './data_xy/x_0004_12.npy',
    './data_xy/x_0005_01.npy',
    './data_xy/x_0005_02.npy',
    './data_xy/x_0005_03.npy',
    './data_xy/x_0005_04.npy',
    './data_xy/x_0005_05.npy',
    './data_xy/x_0005_06.npy',
    './data_xy/x_0005_07.npy',
    './data_xy/x_0005_08.npy',
    './data_xy/x_0005_09.npy',
    './data_xy/x_0005_10.npy',
    './data_xy/x_0005_11.npy',
    './data_xy/x_0005_12.npy',
    './data_xy/x_0006_01.npy',
    './data_xy/x_0006_02.npy',
    './data_xy/x_0006_03.npy',
    './data_xy/x_0006_04.npy',
    './data_xy/x_0006_05.npy',
    './data_xy/x_0006_06.npy',
    './data_xy/x_0006_07.npy',
    './data_xy/x_0006_08.npy',
    './data_xy/x_0006_09.npy',
    './data_xy/x_0006_10.npy',
    './data_xy/x_0006_11.npy',
    './data_xy/x_0006_12.npy',
    './data_xy/x_0007_01.npy',
    './data_xy/x_0007_02.npy',
    './data_xy/x_0007_03.npy',
    './data_xy/x_0007_04.npy',
    './data_xy/x_0007_05.npy',
    './data_xy/x_0007_06.npy',
    './data_xy/x_0007_07.npy',
    './data_xy/x_0007_08.npy',
    './data_xy/x_0007_09.npy',
    './data_xy/x_0007_10.npy',
    './data_xy/x_0007_11.npy',
    './data_xy/x_0007_12.npy',
    './data_xy/x_0008_01.npy',
    './data_xy/x_0008_02.npy',
    './data_xy/x_0008_03.npy',
    './data_xy/x_0008_04.npy',
    './data_xy/x_0008_05.npy',
    './data_xy/x_0008_06.npy'
]
y_files = [
    './data_xy/y_0001_02.npy',
    './data_xy/y_0001_03.npy',
    './data_xy/y_0001_04.npy',
    './data_xy/y_0001_05.npy',
    './data_xy/y_0001_06.npy',
    './data_xy/y_0001_07.npy',
    './data_xy/y_0001_08.npy',
    './data_xy/y_0001_09.npy',
    './data_xy/y_0001_10.npy',
    './data_xy/y_0001_11.npy',
    './data_xy/y_0001_12.npy',
    './data_xy/y_0002_01.npy',
    './data_xy/y_0002_02.npy',
    './data_xy/y_0002_03.npy',
    './data_xy/y_0002_04.npy',
    './data_xy/y_0002_05.npy',
    './data_xy/y_0002_06.npy',
    './data_xy/y_0002_07.npy',
    './data_xy/y_0002_08.npy',
    './data_xy/y_0002_09.npy',
    './data_xy/y_0002_10.npy',
    './data_xy/y_0002_11.npy',
    './data_xy/y_0002_12.npy',
    './data_xy/y_0003_01.npy',
    './data_xy/y_0003_02.npy',
    './data_xy/y_0003_03.npy',
    './data_xy/y_0003_04.npy',
    './data_xy/y_0003_05.npy',
    './data_xy/y_0003_06.npy',
    './data_xy/y_0003_07.npy',
    './data_xy/y_0003_08.npy',
    './data_xy/y_0003_09.npy',
    './data_xy/y_0003_10.npy',
    './data_xy/y_0003_11.npy',
    './data_xy/y_0003_12.npy',
    './data_xy/y_0004_01.npy',
    './data_xy/y_0004_02.npy',
    './data_xy/y_0004_03.npy',
    './data_xy/y_0004_04.npy',
    './data_xy/y_0004_05.npy',
    './data_xy/y_0004_06.npy',
    './data_xy/y_0004_07.npy',
    './data_xy/y_0004_08.npy',
    './data_xy/y_0004_09.npy',
    './data_xy/y_0004_10.npy',
    './data_xy/y_0004_11.npy',
    './data_xy/y_0004_12.npy',
    './data_xy/y_0005_01.npy',
    './data_xy/y_0005_02.npy',
    './data_xy/y_0005_03.npy',
    './data_xy/y_0005_04.npy',
    './data_xy/y_0005_05.npy',
    './data_xy/y_0005_06.npy',
    './data_xy/y_0005_07.npy',
    './data_xy/y_0005_08.npy',
    './data_xy/y_0005_09.npy',
    './data_xy/y_0005_10.npy',
    './data_xy/y_0005_11.npy',
    './data_xy/y_0005_12.npy',
    './data_xy/y_0006_01.npy',
    './data_xy/y_0006_02.npy',
    './data_xy/y_0006_03.npy',
    './data_xy/y_0006_04.npy',
    './data_xy/y_0006_05.npy',
    './data_xy/y_0006_06.npy',
    './data_xy/y_0006_07.npy',
    './data_xy/y_0006_08.npy',
    './data_xy/y_0006_09.npy',
    './data_xy/y_0006_10.npy',
    './data_xy/y_0006_11.npy',
    './data_xy/y_0006_12.npy',
    './data_xy/y_0007_01.npy',
    './data_xy/y_0007_02.npy',
    './data_xy/y_0007_03.npy',
    './data_xy/y_0007_04.npy',
    './data_xy/y_0007_05.npy',
    './data_xy/y_0007_06.npy',
    './data_xy/y_0007_07.npy',
    './data_xy/y_0007_08.npy',
    './data_xy/y_0007_09.npy',
    './data_xy/y_0007_10.npy',
    './data_xy/y_0007_11.npy',
    './data_xy/y_0007_12.npy',
    './data_xy/y_0008_01.npy',
    './data_xy/y_0008_02.npy',
    './data_xy/y_0008_03.npy',
    './data_xy/y_0008_04.npy',
    './data_xy/y_0008_05.npy',
    './data_xy/y_0008_06.npy'
]
print(len(x_files), len(y_files))

os.makedirs('train_numpy_data', exist_ok=True)
os.makedirs('train_numpy_data/samples', exist_ok=True)

# 旧权重
weights = pd.read_parquet('old_test_data/leap-sample-submission.parquet')
del weights['sample_id']
weights = weights.loc[0]
weights = weights.T
weights = weights.to_dict()

weights = np.array(list(weights.values()))
weights = weights.astype(np.float32)
print(weights.shape)

# 归一化参数
mx = np.load('train_numpy_data_kaggle/norm/mx.npy')
sx = np.load('train_numpy_data_kaggle/norm/sx.npy')
my = np.load('train_numpy_data_kaggle/norm/my.npy')
sy = np.load('train_numpy_data_kaggle/norm/sy.npy')

# 采样函数
# 采样函数

def sampling_array(x_filepath, y_filepath, frac=0.2, seed=42):
    
    arr_x = np.load(x_filepath)
    arr_y = np.load(y_filepath)

    x_filename = x_filepath.split('/')[-1].replace('.npy', '')
    y_filename = y_filepath.split('/')[-1].replace('.npy', '')

    print(x_filename, y_filename)
    
    # 随机采样
    sampled_x, _, sampled_y, _ = train_test_split(arr_x, 
                                                  arr_y, 
                                                  test_size=1.-frac, 
                                                  random_state=seed)
    # y 用了新的 weights (0,1)
    # 为了统一, 都用之前的 weights
    # 新的几个 q0002 被置 1 了
    # 但因为后面我们会后处理, 所以不用管
    sampled_y = sampled_y * weights

    # 归一化
    sampled_x = (sampled_x - mx.reshape(1,-1)) / sx.reshape(1,-1)
    sampled_y = (sampled_y - my.reshape(1,-1)) / sy.reshape(1,-1)
    
    # 转 float32
    sampled_x = sampled_x.astype(np.float32)
    sampled_y = sampled_y.astype(np.float32)

    np.save(f'train_numpy_data/samples/{x_filename}', sampled_x)
    np.save(f'train_numpy_data/samples/{y_filename}', sampled_y)

    print(sampled_x.shape, sampled_x.dtype, sampled_y.shape, sampled_y.dtype)


# 全量函数
def sampling_array_full(x_filepath, y_filepath):
    
    sampled_x = np.load(x_filepath)
    sampled_y = np.load(y_filepath)

    x_filename = x_filepath.split('/')[-1].replace('.npy', '')
    y_filename = y_filepath.split('/')[-1].replace('.npy', '')

    print(x_filename, y_filename)
    
    # y 用了新的 weights (0,1)
    # 为了统一, 都用之前的 weights
    # 新的几个 q0002 被置 1 了
    # 但因为后面我们会后处理, 所以不用管
    sampled_y = sampled_y * weights

    # 归一化
    sampled_x = (sampled_x - mx.reshape(1,-1)) / sx.reshape(1,-1)
    sampled_y = (sampled_y - my.reshape(1,-1)) / sy.reshape(1,-1)
    
    # 转 float32
    sampled_x = sampled_x.astype(np.float32)
    sampled_y = sampled_y.astype(np.float32)

    np.save(f'train_numpy_data/samples/{x_filename}', sampled_x)
    np.save(f'train_numpy_data/samples/{y_filename}', sampled_y)

    print(sampled_x.shape, sampled_x.dtype, sampled_y.shape, sampled_y.dtype)
  
# 98% 采样
# for x_filepath, y_filepath in tqdm(zip(x_files, y_files), total=len(x_files)):
#     sampling_array(x_filepath, y_filepath, frac=0.98, seed=71)

for x_filepath, y_filepath in tqdm(zip(x_files, y_files), total=len(x_files)):
    sampling_array_full(x_filepath, y_filepath)
  
