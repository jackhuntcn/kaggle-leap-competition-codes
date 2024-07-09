import warnings
warnings.simplefilter('ignore')
import gc
import glob
from tqdm.auto import tqdm
import numpy as np

x_files = sorted(glob.glob('train_numpy_data/samples/x_*_*.npy'))
print(len(x_files))

all_x_seqs = []
for filepath in tqdm(x_files):
    x_train = np.load(filepath)
    print(filepath, x_train.shape, x_train.dtype)
    train_seqs = np.hstack((
        x_train[:, 0:60].reshape(-1, 1, 60),
        x_train[:, 60:120].reshape(-1, 1, 60),
        x_train[:, 120:180].reshape(-1, 1, 60),
        x_train[:, 180:240].reshape(-1, 1, 60),
        x_train[:, 240:300].reshape(-1, 1, 60),
        x_train[:, 300:360].reshape(-1, 1, 60),
        x_train[:, 360:420].reshape(-1, 1, 60),
        x_train[:, 420:480].reshape(-1, 1, 60),
        x_train[:, 480:540].reshape(-1, 1, 60)
    )).transpose(0, 2, 1)
    all_x_seqs.append(train_seqs)
    del x_train; gc.collect()

x_seqs = np.concatenate(all_x_seqs, axis=0)
print(x_seqs.shape)
del all_x_seqs; gc.collect()
np.save('train_numpy_data/x_seqs', x_seqs)
del x_seqs; gc.collect()

all_x_globals = []
for filepath in tqdm(x_files):
    x_train = np.load(filepath)
    print(filepath, x_train.shape, x_train.dtype)
    train_globals = x_train[:, 540:556]
    all_x_globals.append(train_globals)
    del x_train; gc.collect()
  
x_globals = np.concatenate(all_x_globals, axis=0)
del all_x_globals; gc.collect()
print(x_globals.shape)
np.save('train_numpy_data/x_globals', x_globals)
