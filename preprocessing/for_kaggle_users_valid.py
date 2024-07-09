# import necessary packages

import sys

from climsim_utils.data_utils import *

# set variable names

weights = {'ptend_t_0': 1,
           'ptend_t_1': 1,
           'ptend_t_2': 1,
           'ptend_t_3': 1,
           'ptend_t_4': 1,
           'ptend_t_5': 1,
           'ptend_t_6': 1,
           'ptend_t_7': 1,
           'ptend_t_8': 1,
           'ptend_t_9': 1,
           'ptend_t_10': 1,
           'ptend_t_11': 1,
           'ptend_t_12': 1,
           'ptend_t_13': 1,
           'ptend_t_14': 1,
           'ptend_t_15': 1,
           'ptend_t_16': 1,
           'ptend_t_17': 1,
           'ptend_t_18': 1,
           'ptend_t_19': 1,
           'ptend_t_20': 1,
           'ptend_t_21': 1,
           'ptend_t_22': 1,
           'ptend_t_23': 1,
           'ptend_t_24': 1,
           'ptend_t_25': 1,
           'ptend_t_26': 1,
           'ptend_t_27': 1,
           'ptend_t_28': 1,
           'ptend_t_29': 1,
           'ptend_t_30': 1,
           'ptend_t_31': 1,
           'ptend_t_32': 1,
           'ptend_t_33': 1,
           'ptend_t_34': 1,
           'ptend_t_35': 1,
           'ptend_t_36': 1,
           'ptend_t_37': 1,
           'ptend_t_38': 1,
           'ptend_t_39': 1,
           'ptend_t_40': 1,
           'ptend_t_41': 1,
           'ptend_t_42': 1,
           'ptend_t_43': 1,
           'ptend_t_44': 1,
           'ptend_t_45': 1,
           'ptend_t_46': 1,
           'ptend_t_47': 1,
           'ptend_t_48': 1,
           'ptend_t_49': 1,
           'ptend_t_50': 1,
           'ptend_t_51': 1,
           'ptend_t_52': 1,
           'ptend_t_53': 1,
           'ptend_t_54': 1,
           'ptend_t_55': 1,
           'ptend_t_56': 1,
           'ptend_t_57': 1,
           'ptend_t_58': 1,
           'ptend_t_59': 1,
           'ptend_q0001_0': 0,
           'ptend_q0001_1': 0,
           'ptend_q0001_2': 0,
           'ptend_q0001_3': 0,
           'ptend_q0001_4': 0,
           'ptend_q0001_5': 0,
           'ptend_q0001_6': 0,
           'ptend_q0001_7': 0,
           'ptend_q0001_8': 0,
           'ptend_q0001_9': 0,
           'ptend_q0001_10': 0,
           'ptend_q0001_11': 0,
           'ptend_q0001_12': 1,
           'ptend_q0001_13': 1,
           'ptend_q0001_14': 1,
           'ptend_q0001_15': 1,
           'ptend_q0001_16': 1,
           'ptend_q0001_17': 1,
           'ptend_q0001_18': 1,
           'ptend_q0001_19': 1,
           'ptend_q0001_20': 1,
           'ptend_q0001_21': 1,
           'ptend_q0001_22': 1,
           'ptend_q0001_23': 1,
           'ptend_q0001_24': 1,
           'ptend_q0001_25': 1,
           'ptend_q0001_26': 1,
           'ptend_q0001_27': 1,
           'ptend_q0001_28': 1,
           'ptend_q0001_29': 1,
           'ptend_q0001_30': 1,
           'ptend_q0001_31': 1,
           'ptend_q0001_32': 1,
           'ptend_q0001_33': 1,
           'ptend_q0001_34': 1,
           'ptend_q0001_35': 1,
           'ptend_q0001_36': 1,
           'ptend_q0001_37': 1,
           'ptend_q0001_38': 1,
           'ptend_q0001_39': 1,
           'ptend_q0001_40': 1,
           'ptend_q0001_41': 1,
           'ptend_q0001_42': 1,
           'ptend_q0001_43': 1,
           'ptend_q0001_44': 1,
           'ptend_q0001_45': 1,
           'ptend_q0001_46': 1,
           'ptend_q0001_47': 1,
           'ptend_q0001_48': 1,
           'ptend_q0001_49': 1,
           'ptend_q0001_50': 1,
           'ptend_q0001_51': 1,
           'ptend_q0001_52': 1,
           'ptend_q0001_53': 1,
           'ptend_q0001_54': 1,
           'ptend_q0001_55': 1,
           'ptend_q0001_56': 1,
           'ptend_q0001_57': 1,
           'ptend_q0001_58': 1,
           'ptend_q0001_59': 1,
           'ptend_q0002_0': 0,
           'ptend_q0002_1': 0,
           'ptend_q0002_2': 0,
           'ptend_q0002_3': 0,
           'ptend_q0002_4': 0,
           'ptend_q0002_5': 0,
           'ptend_q0002_6': 0,
           'ptend_q0002_7': 0,
           'ptend_q0002_8': 0,
           'ptend_q0002_9': 0,
           'ptend_q0002_10': 0,
           'ptend_q0002_11': 0,
           'ptend_q0002_12': 1,
           'ptend_q0002_13': 1,
           'ptend_q0002_14': 1,
           'ptend_q0002_15': 1,
           'ptend_q0002_16': 1,
           'ptend_q0002_17': 1,
           'ptend_q0002_18': 1,
           'ptend_q0002_19': 1,
           'ptend_q0002_20': 1,
           'ptend_q0002_21': 1,
           'ptend_q0002_22': 1,
           'ptend_q0002_23': 1,
           'ptend_q0002_24': 1,
           'ptend_q0002_25': 1,
           'ptend_q0002_26': 1,
           'ptend_q0002_27': 1,
           'ptend_q0002_28': 1,
           'ptend_q0002_29': 1,
           'ptend_q0002_30': 1,
           'ptend_q0002_31': 1,
           'ptend_q0002_32': 1,
           'ptend_q0002_33': 1,
           'ptend_q0002_34': 1,
           'ptend_q0002_35': 1,
           'ptend_q0002_36': 1,
           'ptend_q0002_37': 1,
           'ptend_q0002_38': 1,
           'ptend_q0002_39': 1,
           'ptend_q0002_40': 1,
           'ptend_q0002_41': 1,
           'ptend_q0002_42': 1,
           'ptend_q0002_43': 1,
           'ptend_q0002_44': 1,
           'ptend_q0002_45': 1,
           'ptend_q0002_46': 1,
           'ptend_q0002_47': 1,
           'ptend_q0002_48': 1,
           'ptend_q0002_49': 1,
           'ptend_q0002_50': 1,
           'ptend_q0002_51': 1,
           'ptend_q0002_52': 1,
           'ptend_q0002_53': 1,
           'ptend_q0002_54': 1,
           'ptend_q0002_55': 1,
           'ptend_q0002_56': 1,
           'ptend_q0002_57': 1,
           'ptend_q0002_58': 1,
           'ptend_q0002_59': 1,
           'ptend_q0003_0': 0,
           'ptend_q0003_1': 0,
           'ptend_q0003_2': 0,
           'ptend_q0003_3': 0,
           'ptend_q0003_4': 0,
           'ptend_q0003_5': 0,
           'ptend_q0003_6': 0,
           'ptend_q0003_7': 0,
           'ptend_q0003_8': 0,
           'ptend_q0003_9': 0,
           'ptend_q0003_10': 0,
           'ptend_q0003_11': 0,
           'ptend_q0003_12': 1,
           'ptend_q0003_13': 1,
           'ptend_q0003_14': 1,
           'ptend_q0003_15': 1,
           'ptend_q0003_16': 1,
           'ptend_q0003_17': 1,
           'ptend_q0003_18': 1,
           'ptend_q0003_19': 1,
           'ptend_q0003_20': 1,
           'ptend_q0003_21': 1,
           'ptend_q0003_22': 1,
           'ptend_q0003_23': 1,
           'ptend_q0003_24': 1,
           'ptend_q0003_25': 1,
           'ptend_q0003_26': 1,
           'ptend_q0003_27': 1,
           'ptend_q0003_28': 1,
           'ptend_q0003_29': 1,
           'ptend_q0003_30': 1,
           'ptend_q0003_31': 1,
           'ptend_q0003_32': 1,
           'ptend_q0003_33': 1,
           'ptend_q0003_34': 1,
           'ptend_q0003_35': 1,
           'ptend_q0003_36': 1,
           'ptend_q0003_37': 1,
           'ptend_q0003_38': 1,
           'ptend_q0003_39': 1,
           'ptend_q0003_40': 1,
           'ptend_q0003_41': 1,
           'ptend_q0003_42': 1,
           'ptend_q0003_43': 1,
           'ptend_q0003_44': 1,
           'ptend_q0003_45': 1,
           'ptend_q0003_46': 1,
           'ptend_q0003_47': 1,
           'ptend_q0003_48': 1,
           'ptend_q0003_49': 1,
           'ptend_q0003_50': 1,
           'ptend_q0003_51': 1,
           'ptend_q0003_52': 1,
           'ptend_q0003_53': 1,
           'ptend_q0003_54': 1,
           'ptend_q0003_55': 1,
           'ptend_q0003_56': 1,
           'ptend_q0003_57': 1,
           'ptend_q0003_58': 1,
           'ptend_q0003_59': 1,
           'ptend_u_0': 0,
           'ptend_u_1': 0,
           'ptend_u_2': 0,
           'ptend_u_3': 0,
           'ptend_u_4': 0,
           'ptend_u_5': 0,
           'ptend_u_6': 0,
           'ptend_u_7': 0,
           'ptend_u_8': 0,
           'ptend_u_9': 0,
           'ptend_u_10': 0,
           'ptend_u_11': 0,
           'ptend_u_12': 1,
           'ptend_u_13': 1,
           'ptend_u_14': 1,
           'ptend_u_15': 1,
           'ptend_u_16': 1,
           'ptend_u_17': 1,
           'ptend_u_18': 1,
           'ptend_u_19': 1,
           'ptend_u_20': 1,
           'ptend_u_21': 1,
           'ptend_u_22': 1,
           'ptend_u_23': 1,
           'ptend_u_24': 1,
           'ptend_u_25': 1,
           'ptend_u_26': 1,
           'ptend_u_27': 1,
           'ptend_u_28': 1,
           'ptend_u_29': 1,
           'ptend_u_30': 1,
           'ptend_u_31': 1,
           'ptend_u_32': 1,
           'ptend_u_33': 1,
           'ptend_u_34': 1,
           'ptend_u_35': 1,
           'ptend_u_36': 1,
           'ptend_u_37': 1,
           'ptend_u_38': 1,
           'ptend_u_39': 1,
           'ptend_u_40': 1,
           'ptend_u_41': 1,
           'ptend_u_42': 1,
           'ptend_u_43': 1,
           'ptend_u_44': 1,
           'ptend_u_45': 1,
           'ptend_u_46': 1,
           'ptend_u_47': 1,
           'ptend_u_48': 1,
           'ptend_u_49': 1,
           'ptend_u_50': 1,
           'ptend_u_51': 1,
           'ptend_u_52': 1,
           'ptend_u_53': 1,
           'ptend_u_54': 1,
           'ptend_u_55': 1,
           'ptend_u_56': 1,
           'ptend_u_57': 1,
           'ptend_u_58': 1,
           'ptend_u_59': 1,
           'ptend_v_0': 0,
           'ptend_v_1': 0,
           'ptend_v_2': 0,
           'ptend_v_3': 0,
           'ptend_v_4': 0,
           'ptend_v_5': 0,
           'ptend_v_6': 0,
           'ptend_v_7': 0,
           'ptend_v_8': 0,
           'ptend_v_9': 0,
           'ptend_v_10': 0,
           'ptend_v_11': 0,
           'ptend_v_12': 1,
           'ptend_v_13': 1,
           'ptend_v_14': 1,
           'ptend_v_15': 1,
           'ptend_v_16': 1,
           'ptend_v_17': 1,
           'ptend_v_18': 1,
           'ptend_v_19': 1,
           'ptend_v_20': 1,
           'ptend_v_21': 1,
           'ptend_v_22': 1,
           'ptend_v_23': 1,
           'ptend_v_24': 1,
           'ptend_v_25': 1,
           'ptend_v_26': 1,
           'ptend_v_27': 1,
           'ptend_v_28': 1,
           'ptend_v_29': 1,
           'ptend_v_30': 1,
           'ptend_v_31': 1,
           'ptend_v_32': 1,
           'ptend_v_33': 1,
           'ptend_v_34': 1,
           'ptend_v_35': 1,
           'ptend_v_36': 1,
           'ptend_v_37': 1,
           'ptend_v_38': 1,
           'ptend_v_39': 1,
           'ptend_v_40': 1,
           'ptend_v_41': 1,
           'ptend_v_42': 1,
           'ptend_v_43': 1,
           'ptend_v_44': 1,
           'ptend_v_45': 1,
           'ptend_v_46': 1,
           'ptend_v_47': 1,
           'ptend_v_48': 1,
           'ptend_v_49': 1,
           'ptend_v_50': 1,
           'ptend_v_51': 1,
           'ptend_v_52': 1,
           'ptend_v_53': 1,
           'ptend_v_54': 1,
           'ptend_v_55': 1,
           'ptend_v_56': 1,
           'ptend_v_57': 1,
           'ptend_v_58': 1,
           'ptend_v_59': 1,
           'cam_out_NETSW': 1,
           'cam_out_FLWDS': 1,
           'cam_out_PRECSC': 1,
           'cam_out_PRECC': 1,
           'cam_out_SOLS': 1,
           'cam_out_SOLL': 1,
           'cam_out_SOLSD': 1,
           'cam_out_SOLLD': 1}

v2_inputs = ['state_t',
             'state_q0001',
             'state_q0002',
             'state_q0003',
             'state_u',
             'state_v',
             'state_ps',
             'pbuf_SOLIN',
             'pbuf_LHFLX',
             'pbuf_SHFLX',
             'pbuf_TAUX',
             'pbuf_TAUY',
             'pbuf_COSZRS',
             'cam_in_ALDIF',
             'cam_in_ALDIR',
             'cam_in_ASDIF',
             'cam_in_ASDIR',
             'cam_in_LWUP',
             'cam_in_ICEFRAC',
             'cam_in_LANDFRAC',
             'cam_in_OCNFRAC',
             'cam_in_SNOWHICE',
             'cam_in_SNOWHLAND',
             'pbuf_ozone', # outside of the upper troposphere lower stratosphere (UTLS, corresponding to indices 5-21), variance in minimal for these last 3 
             'pbuf_CH4',
             'pbuf_N2O']

v2_outputs = ['ptend_t',
              'ptend_q0001',
              'ptend_q0002',
              'ptend_q0003',
              'ptend_u',
              'ptend_v',
              'cam_out_NETSW',
              'cam_out_FLWDS',
              'cam_out_PRECSC',
              'cam_out_PRECC',
              'cam_out_SOLS',
              'cam_out_SOLL',
              'cam_out_SOLSD',
              'cam_out_SOLLD']

vertically_resolved = ['state_t', 
                       'state_q0001', 
                       'state_q0002', 
                       'state_q0003', 
                       'state_u', 
                       'state_v', 
                       'pbuf_ozone', 
                       'pbuf_CH4', 
                       'pbuf_N2O', 
                       'ptend_t', 
                       'ptend_q0001', 
                       'ptend_q0002', 
                       'ptend_q0003', 
                       'ptend_u', 
                       'ptend_v']

ablated_vars = ['ptend_q0001',
                'ptend_q0002',
                'ptend_q0003',
                'ptend_u',
                'ptend_v']

v2_vars = v2_inputs + v2_outputs

train_col_names = []
ablated_col_names = []
for var in v2_vars:
    if var in vertically_resolved:
        for i in range(60):
            train_col_names.append(var + '_' + str(i))
            if i < 12 and var in ablated_vars:
                ablated_col_names.append(var + '_' + str(i))
    else:
        train_col_names.append(var)

input_col_names = []
for var in v2_inputs:
    if var in vertically_resolved:
        for i in range(60):
            input_col_names.append(var + '_' + str(i))
    else:
        input_col_names.append(var)

output_col_names = []
for var in v2_outputs:
    if var in vertically_resolved:
        for i in range(60):
            output_col_names.append(var + '_' + str(i))
    else:
        output_col_names.append(var)

assert(len(train_col_names) == 17 + 60*9 + 60*6 + 8)
assert(len(input_col_names) == 17 + 60*9)
assert(len(output_col_names) == 60*6 + 8)
assert(len(set(output_col_names).intersection(set(ablated_col_names))) == len(ablated_col_names))

# initialize data_utils object

grid_path = './grid_info/ClimSim_low-res_grid-info.nc'
norm_path = './preprocessing/normalizations/'

grid_info = xr.open_dataset(grid_path)
input_mean = None
input_max = None
input_min = None
output_scale = None

data = data_utils(grid_info = grid_info, 
                  input_mean = input_mean, 
                  input_max = input_max, 
                  input_min = input_min, 
                  output_scale = output_scale)

data.set_to_v2_vars()

# do not normalize
data.normalize = False

# create training data

# set data path for training data
data.data_path = '/root/autodl-tmp/train/'

# set regular expressions for selecting training data
year = str(sys.argv[1])
month = str(sys.argv[2])

data.set_regexps(data_split = 'train', 
                 regexps = [f'E3SM-MMF.mli.{year}-{month}-*-*.nc'])

# set temporal subsampling
data.set_stride_sample(data_split = 'train', stride_sample = 1)

# create list of files to extract data from
data.set_filelist(data_split = 'train')

# save numpy files of training data
data_loader = data.load_ncdata_with_generator(data_split = 'train')
npy_iterator = list(data_loader.as_numpy_iterator())
npy_input = np.concatenate([npy_iterator[x][0] for x in range(len(npy_iterator))])
npy_output = np.concatenate([npy_iterator[x][1] for x in range(len(npy_iterator))])
train_npy = np.concatenate([npy_input, npy_output], axis = 1)
train_index = ["train_" + str(x) for x in range(train_npy.shape[0])]

train = pd.DataFrame(train_npy, index = train_index, columns = train_col_names)
train.index.name = 'sample_id'
print('dropping cam_in_SNOWHICE because of strange values')
train.drop('cam_in_SNOWHICE', axis=1, inplace=True)

# ASSERT, SHAPE, CSV, PRINT
assert sum(train.isnull().any()) == 0
print(train.shape)
train.to_parquet('./new_valid_data/train_{year}_{month}.parquet')
print('finished creating train data')
