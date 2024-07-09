import warnings
warnings.simplefilter('ignore')
import gc
import glob
from tqdm.auto import tqdm
import numpy as np

y_files = sorted(glob.glob('train_numpy_data/samples/y_*_*.npy'))
print(len(y_files))

all_y = []
for filepath in tqdm(y_files):
    y_train = np.load(filepath)
    print(filepath, y_train.shape, y_train.dtype)
    all_y.append(y_train)
    del y_train; gc.collect()

y_train = np.concatenate(all_y, axis=0)
del all_y; gc.collect()
print(y_train.shape)
np.save('train_numpy_data/y_train', y_train)
