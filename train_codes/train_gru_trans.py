import warnings
warnings.simplefilter('ignore')

import gc
import os
import time
import torch
import random
import datetime
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
from torch.utils.data import Dataset, DataLoader
from transformers import get_cosine_schedule_with_warmup

N_TARGETS = 368
BATCH_SIZE = 5120
SCHEDULER_PATIENCE = 2
SCHEDULER_FACTOR = 10**(-0.5)
EPOCHS = 30
PATIENCE = 5
PRINT_FREQ = 1000

def format_time(elapsed):
    """Take a time in seconds and return a string hh:mm:ss."""
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def seed_everything(seed_val=1325):
    """Seed everything."""
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

seed_everything()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

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

start_time = time.time()
x_train_seq = np.load('train_numpy_data/x_seqs.npy')
x_train_global = np.load('train_numpy_data/x_globals.npy')
y_train = np.load('train_numpy_data/y_train.npy')
print(x_train_seq.shape)
print(x_train_global.shape)
print(y_train.shape)
used_time = (time.time() - start_time) / 60
print(f'loaded train data used time: {used_time:.2f} minutes')

start_time = time.time()
x_valid_seq = np.load('new_valid_data/x_seqs_valid.npy')
x_valid_global = np.load('new_valid_data/x_globals_valid.npy')
y_valid = np.load('new_valid_data/y_valid.npy')
print(x_valid_seq.shape)
print(x_valid_global.shape)
print(y_valid.shape)
used_time = (time.time() - start_time) / 60
print(f'loaded valid data used time: {used_time:.2f} minutes')


df_valid = pd.read_parquet('new_valid_data/valid.parquet')
weights = pd.read_parquet('old_test_data/leap-sample-submission.parquet')
del weights['sample_id']
weights = weights.loc[0]
weights = weights.T
weights = weights.to_dict()
for target in TARGETCOLS:
    df_valid[target] = df_valid[target] * weights[target]

sy = np.load('train_numpy_data_kaggle/norm/sy.npy')
my = np.load('train_numpy_data_kaggle/norm/my.npy')


class NumpyDataset(Dataset):
    def __init__(self, x_seq, x_global, y):
        """
        Initialize with NumPy arrays.
        """
        self.x_seq = x_seq
        self.x_global = x_global
        self.y = y

    def __len__(self):
        """
        Total number of samples.
        """
        return self.x_seq.shape[0]

    def __getitem__(self, index):
        """
        Generate one sample of data.
        """
        # Convert the data to tensors when requested

        arr_1 = self.x_seq[index]                                              # (60,9)
        arr_2 = self.x_global[index].repeat(60).reshape(-1,60).transpose(1,0)  # (60,16)
        arr_x = np.hstack((arr_1, arr_2))                                      # (60,25)

        x = torch.from_numpy(arr_x).float()
        y = torch.from_numpy(self.y[index]).float()

        return x, y



train_dataset = NumpyDataset(x_train_seq, x_train_global, y_train)
valid_dataset = NumpyDataset(x_valid_seq, x_valid_global, y_valid)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=8)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=8)


class LeapModel(nn.Module):
    def __init__(self,
                 input_size,
                 seq_len,
                 hidden_size,
                 output_size,
                 num_layers=1,
                 bidirectional=False,
                 dropout=0.3,
                 hidden_layers=[128, 256],
                 nhead=8,
                 num_transformer_layers=2):

        super().__init__()
        self.input_size = input_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.output_size = output_size

        # LSTM layer
        self.rnn =  nn.GRU(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           bidirectional=bidirectional,
                           batch_first=True,
                           dropout=dropout)

        # Transformer layer
        transformer_input_size = hidden_size * 2 if bidirectional else hidden_size
        self.transformer_layer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=transformer_input_size, nhead=nhead, dropout=dropout),
            num_layers=num_transformer_layers
        )

        # Fully connected layers
        if hidden_layers and len(hidden_layers):
            first_layer = nn.Linear(transformer_input_size, hidden_layers[0])
            self.hidden_layers = nn.ModuleList(
                [first_layer] + \
                [nn.Linear(hidden_layers[i], hidden_layers[i+1]) for i in range(len(hidden_layers) - 1)]
            )
            for layer in self.hidden_layers:
                nn.init.kaiming_normal_(layer.weight.data)
            self.intermediate_layer = nn.Linear(hidden_layers[-1], self.input_size)
            self.output_layer = nn.Linear(hidden_layers[-1], output_size)
            nn.init.kaiming_normal_(self.output_layer.weight.data)
        else:
            self.hidden_layers = []
            self.intermediate_layer = nn.Linear(transformer_input_size, self.input_size)
            self.output_layer = nn.Linear(transformer_input_size, output_size)
            nn.init.kaiming_normal_(self.output_layer.weight.data)

        self.activation_fn = torch.nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # LSTM layer
        lstm_output, _ = self.rnn(x)
        
        # Transformer layer
        transformer_output = self.transformer_layer(lstm_output)
        
        # Apply dropout and activation
        x = self.dropout(self.activation_fn(transformer_output))
        
        # Fully connected layers
        for hidden_layer in self.hidden_layers:
            x = self.activation_fn(hidden_layer(x))
            x = self.dropout(x)
        x = self.output_layer(x)

        # Reshape output
        o_s = x[:, :, :6]
        o_s = o_s.permute(0, 2, 1).reshape(-1, 360)
        o_g = x[:, :, 6:]
        o_g = o_g.mean(dim=1)
        out = torch.cat([o_s, o_g], dim=1)

        return out


input_size = 25
output_size = 14
seq_len = 60

hidden_size = 256
hidden_layers = [256, 512]
num_layers = 6
dropout = 0.1
nhead = 8
num_transformer_layers = 1

model = LeapModel(
    input_size=input_size,
    seq_len=seq_len,
    hidden_size=hidden_size,
    output_size=output_size,
    num_layers=num_layers,
    hidden_layers=hidden_layers,
    dropout=dropout,
    bidirectional=True,
    nhead=nhead,
    num_transformer_layers=num_transformer_layers
).to(device)

criterion = nn.SmoothL1Loss(reduction='mean')
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
num_train_steps = int(y_train.shape[0] / BATCH_SIZE * EPOCHS)
scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=0, num_training_steps=num_train_steps, num_cycles=0.5)

model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
print(model)

def calc_r2_p(valid_pred):
    valid_pred = valid_pred.cpu().numpy()
    valid_pred = valid_pred * sy.reshape(1,-1) + my.reshape(1,-1)
    df_pred = pd.DataFrame(valid_pred)
    df_pred.columns = TARGETCOLS
    targets_unpredictable = []
    for target in weights:
        if weights[target] == 0.:
            targets_unpredictable.append(target)
    for target in targets_unpredictable:
        df_pred[target] = 0.
    for target in [f'ptend_q0002_{i}' for i in range(12, 28)]:
        df_pred[target] = -df_valid[target.replace("ptend", "state")] * weights[target] / 1200.
    score = r2_score(df_valid[TARGETCOLS].values, df_pred[TARGETCOLS].values)
    return score

ts = time.time()

best_val_loss = float('inf')
best_val_score = -1
best_model_state = None
patience_count = 0
for epoch in range(EPOCHS):
    print("")
    epoch_start_time = time.time()
    model.train()
    total_loss = 0
    steps = 0
    for batch_idx, (x, labels) in enumerate(train_loader):
        x = x.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        steps += 1

        if (batch_idx + 1) % PRINT_FREQ == 0:
            current_lr = optimizer.param_groups[0]['lr']
            elapsed_time = format_time(time.time() - ts)
            print(f'  Epoch: {epoch+1}',\
                  f'  Batch: {batch_idx + 1}/{len(train_loader)}',\
                  f'  Train Loss: {total_loss / steps:.4f}',\
                  f'  LR: {current_lr:.3e}',\
                  f'  Time: {elapsed_time}', flush=True)
            total_loss = 0
            steps = 0

        scheduler.step()

    model.eval()
    val_loss = 0
    y_true = torch.tensor([], device=device)
    all_outputs = torch.tensor([], device=device)
    with torch.no_grad():
        for x, labels in valid_loader:
            x = x.to(device)
            labels = labels.to(device)
            outputs = model(x)
            val_loss += criterion(outputs, labels).item()
            y_true = torch.cat((y_true, labels), 0)
            all_outputs = torch.cat((all_outputs, outputs), 0)

    r2 = calc_r2_p(all_outputs)
    avg_val_loss = val_loss / len(valid_loader)

    print(f'\nEpoch: {epoch+1}  Val Loss: {avg_val_loss:.4f} R2 score: {r2:.4f}')

    if r2 > best_val_score:
        best_val_score = r2
        best_model_state = model.state_dict()
        patience_count = 0
        print("Saving new best model and resetting patience counter.")
        torch.save(best_model_state, 'best_model_weights.pth')
    else:
        patience_count += 1
        print(f"No improvement in validation loss for {patience_count} epochs.")

    if patience_count >= PATIENCE:
        print("Stopping early due to no improvement in validation loss.")
        break

    epoch_used_time = (time.time() - epoch_start_time) / 60
    print(f"Epoch elapsed time: {epoch_used_time:.2f} minutes")
