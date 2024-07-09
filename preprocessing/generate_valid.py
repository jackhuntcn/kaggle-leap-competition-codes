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

data_paths = sorted(glob.glob('./new_valid_data/train_*.parquet'))
print(data_paths)

valid_data = []

for path in tqdm(data_paths):
    data = pd.read_parquet(path)
    data = data.sample(frac=0.105, random_state=42).reset_index(drop=True)
    valid_data.append(data)

valid_data = pd.concat(valid_data).reset_index(drop=True)
print(valid_data)

valid_data.to_parquet('./new_valid_data/valid.parquet')

# set columns
STATE_T = [f'state_t_{i}' for i in range(60)]           # 0:60
STATE_Q0001 = [f'state_q0001_{i}' for i in range(60)]   # 60:120
STATE_Q0002 = [f'state_q0002_{i}' for i in range(60)]   # 120:180
STATE_Q0003 = [f'state_q0003_{i}' for i in range(60)]   # 180:240
STATE_U = [f'state_u_{i}' for i in range(60)]           # 240:300
STATE_V = [f'state_v_{i}' for i in range(60)]           # 300:360
PBUF_OZONE = [f'pbuf_ozone_{i}' for i in range(60)]     # 360:420
PBUF_CH4 = [f'pbuf_CH4_{i}' for i in range(60)]         # 420:480
PBUF_N2O = [f'pbuf_N2O_{i}' for i in range(60)]         # 480:540
GLOBALS = [
    'state_ps','pbuf_SOLIN','pbuf_LHFLX','pbuf_SHFLX','pbuf_TAUX','pbuf_TAUY',
    'pbuf_COSZRS','cam_in_ALDIF','cam_in_ALDIR','cam_in_ASDIF','cam_in_ASDIR',
    'cam_in_LWUP','cam_in_ICEFRAC','cam_in_LANDFRAC','cam_in_OCNFRAC','cam_in_SNOWHLAND'
]                                                       # 540:556
FEATCOLS = STATE_T + STATE_Q0001 + STATE_Q0002 + STATE_Q0003 + STATE_U + STATE_V +\
           PBUF_OZONE + PBUF_CH4 + PBUF_N2O + GLOBALS
print(len(FEATCOLS))

TARGETCOLS = [f'ptend_t_{i}' for i in range(60)] +\
             [f'ptend_q0001_{i}' for i in range(60)] +\
             [f'ptend_q0002_{i}' for i in range(60)] +\
             [f'ptend_q0003_{i}' for i in range(60)] +\
             [f'ptend_u_{i}' for i in range(60)] +\
             [f'ptend_v_{i}' for i in range(60)] +\
             ['cam_out_NETSW','cam_out_FLWDS','cam_out_PRECSC',
              'cam_out_PRECC','cam_out_SOLS','cam_out_SOLL',
              'cam_out_SOLSD','cam_out_SOLLD']
print(len(TARGETCOLS))

# X 正则化
mx = np.load('train_numpy_data_kaggle/norm/mx.npy')
sx = np.load('train_numpy_data_kaggle/norm/sx.npy')

x = valid_data[FEATCOLS].values.astype(np.float32)
x = (x - mx.reshape(1,-1)) / sx.reshape(1,-1)
x = x.astype(np.float32)

x_seqs = np.hstack((
    x[:, 0:60].reshape(-1, 1, 60),
    x[:, 60:120].reshape(-1, 1, 60),
    x[:, 120:180].reshape(-1, 1, 60),
    x[:, 180:240].reshape(-1, 1, 60),
    x[:, 240:300].reshape(-1, 1, 60),
    x[:, 300:360].reshape(-1, 1, 60),
    x[:, 360:420].reshape(-1, 1, 60),
    x[:, 420:480].reshape(-1, 1, 60),
    x[:, 480:540].reshape(-1, 1, 60)
)).transpose(0, 2, 1)
print(x_seqs.shape)

x_globals = x[:, 540:556]
print(x_globals.shape)

# 旧权重
weights = pd.read_parquet('old_test_data/leap-sample-submission.parquet')
del weights['sample_id']
weights = weights.loc[0]
weights = weights.T
weights = weights.to_dict()

weights = np.array(list(weights.values()))
weights = weights.astype(np.float32)
print(weights.shape)

# y 归一化
my = np.load('train_numpy_data_kaggle/norm/my.npy')
sy = np.load('train_numpy_data_kaggle/norm/sy.npy')

y = valid_data[TARGETCOLS].values.astype(np.float32)
# 为了统一, 都用之前的 weights
y = y * weights
# 然后再归一化
y = (y - my.reshape(1,-1)) / sy.reshape(1,-1)
y = y.astype(np.float32)
print(y.shape)

np.save('new_valid_data/x_seqs_valid', x_seqs)
np.save('new_valid_data/x_globals_valid', x_globals)
np.save('new_valid_data/y_valid', y)
