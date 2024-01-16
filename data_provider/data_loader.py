import math
import os
import warnings
from collections import namedtuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset

from utils.timefeatures import time_features

warnings.filterwarnings('ignore')


DatasetItem = namedtuple(
    'DatasetItem', [
        'prev_x', 'prev_y', 'prev_mark',
        'target_x', 'target_y', 'target_mark'
    ]
)


ScalerSelector = {
    "minmax": MinMaxScaler(),
    "std": StandardScaler(),
}


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 **kwargs
                 ):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'valid']
        type_map = {'train': 0, 'valid': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(
            self, root_path, flag='train', size=None,
            features='S', data_path='ETTm1.csv',
            target='OT', scale=True, timeenc=0, freq='t',
            **kwargs
    ):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'valid']
        type_map = {'train': 0, 'valid': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


"""
class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_valid = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_valid, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
"""


class Dataset_Custom(Dataset):
    def __init__(
            self, root_path, flag='train', size=None,
            features='S', data_path='ETTh1.csv',
            target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None,
            **kwargs
    ):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'valid']
        # type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = flag

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.lag = kwargs.get('lag', 0)
        # if self.lag > 0:
        #     assert self.features == "S"

        self.__read_data__()


    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        # cols = list(df_raw.columns);
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_valid = len(df_raw) - num_train - num_test
        # border1s = [        0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        # border2s = [num_train,     num_train + num_valid, len(df_raw)]

        # border1, border2 = border1s[self.set_type], border2s[self.set_type]
        data_split = {
            "train": {"start": 0, "end": num_train},
            "valid": {"start": num_train - self.seq_len - self.lag, "end": num_train + num_valid},
            "test": {"start": len(df_raw) - num_test - self.seq_len - self.lag, "end": len(df_raw)},
        }
        border1, border2 = data_split[self.set_type]['start'], data_split[self.set_type]['end']

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            # train_data = df_data[border1s[0]:border2s[0]]
            train_data = df_data[data_split['train']['start']:data_split['train']['end']]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # df_stamp = df_raw[['date']][border1:border2]
        # df_stamp['date'] = pd.to_datetime(df_stamp.date)
        # data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin, s_end = index, index + self.seq_len
        r_begin, r_end = s_end - self.label_len + self.lag, s_end + self.pred_len + self.lag
        if self.lag > 0:
            seq_x = []
            seq_x_mark = []
            for i in range(self.lag):
                seq_x.append(self.data_x[s_begin+i:s_end+i].squeeze())
                seq_x_mark.append(self.data_stamp[s_begin+i:s_end+i].squeeze())
            seq_x = np.array(seq_x).reshape(self.seq_len, -1)
            seq_x_mark = np.array(seq_x_mark).reshape(self.seq_len, -1)
        else:
            seq_x = self.data_x[s_begin:s_end]
            seq_x_mark = self.data_stamp[s_begin:s_end]
        # select y.
        seq_y_mark = self.data_stamp[r_begin:r_end]
        seq_y = self.data_y[r_begin:r_end]
        if self.inverse:
            seq_y = np.concatenate(
                [self.data_x[r_begin:r_begin + self.label_len], self.data_y[r_begin + self.label_len:r_end]], 0)
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1 - self.lag

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None,
                 **kwargs
                 ):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        else:
            raise ValueError

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        else:
            raise ValueError

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Gefcom(Dataset):
    def __init__(
            self, root_path, flag='train', size=None,
            features='S', data_path='ETTh1.csv',
            target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None,
            scaler="std",
            **kwargs
    ):
        # size [seq_len, label_len, pred_len]
        # info
        if size is None:
            self.seq_len = 24 * 12
            self.label_len = 24 * 6
            self.pred_len = 24 * 31
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'valid']

        self.set_type = flag
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.scaler = ScalerSelector.get(scaler, ScalerSelector['std'])

        self.lag = kwargs.get('lag', 0)
        self.zone = kwargs.get('zone', 'CT')
        self.real_time = kwargs.get('real_time', False)
        # if self.lag > 0:
        #     assert self.features == "S"
        self.__read_data__()

    def set_pred_start(self, pred_start):
        self.pred_start = pred_start


    def __read_data__(self):

        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        # all_data = df_raw[['ts', 'zone', 'demand', 'drybulb', 'dewpnt', 'holiday_name', 'holiday']]
        df_raw = df_raw[['ts', 'zone', 'demand', 'drybulb', 'dewpnt', ]]
        df_raw.rename(columns={"ts": "date"}, inplace=True)
        # df_raw = df_raw.groupby('zone').get_group(self.zone)
        df_raw = df_raw.loc[df_raw['zone'] == self.zone].drop(columns=['zone']).sort_values(by='date')


        df_raw['date'] = pd.to_datetime(df_raw['date'])
        train_start, train_end = pd.Timestamp(2013, 1, 1), pd.Timestamp(2014, 7, 1)
        valid_end, test_end = pd.Timestamp(2015, 1, 1), pd.Timestamp(2015, 12, 31)
        train_data = df_raw.loc[
            # df_raw['date'] < pd.Timestamp(2016, 7, 1)
            df_raw['date'].apply(lambda x: train_start <= x < train_end)
        ]
        valid_data = df_raw.loc[
            df_raw['date'].apply(lambda x: train_end <= x < valid_end)
        ]
        test_data = df_raw.loc[
            # pd.Timestamp(2016, 10, 1) <= df_raw['date']
            df_raw['date'].apply(lambda x: valid_end <= x < test_end)
        ]
        train_start = sum(df_raw['date'] < train_start)
        valid_start = sum(df_raw['date'] < train_end)
        test_start = sum(df_raw['date'] < valid_end)

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        # cols = list(df_raw.columns);
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]

        num_train, num_valid, num_test = len(train_data), len(valid_data), len(test_data)
        data_split = {
            "train": {
                "start": train_start,
                "end": train_start + num_train
            },
            "valid": {
                "start": valid_start - self.seq_len - self.lag,
                "end": valid_start + num_valid
            },
            "test": {
                "start": test_start - self.seq_len - self.lag,
                "end": test_start + num_test
            },
        }
        border1, border2 = data_split[self.set_type]['start'], data_split[self.set_type]['end']
        if self.real_time:
            if self.set_type == 'train':
                pred_start = train_start
            elif self.set_type == 'valid':
                pred_start = valid_start
            else:
                pred_start = test_start
            pred_start -= data_split[self.set_type]['start']
            self.set_pred_start(pred_start=pred_start)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        else:
            raise ValueError

        if self.scale:
            # train_data = df_data[border1s[0]:border2s[0]]
            train_data = df_data[data_split['train']['start']:data_split['train']['end']]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
            # TODO(rzhao): inverse_transform the prediction. (done)
        else:
            data = df_data.values

        # df_stamp = df_raw[['date']][border1:border2]
        # df_stamp['date'] = pd.to_datetime(df_stamp.date)
        # data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        else:
            raise ValueError

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        if self.real_time:
            if index+1 == self.__len__():
                r_begin = len(self.data_x) - self.pred_len
                r_end = len(self.data_x)
            else:
                r_begin = self.pred_start + index * self.pred_len
                r_end = r_begin + self.pred_len
            s_begin = r_begin - self.seq_len - self.lag
            s_end = s_begin + self.seq_len
        else:
            s_begin, s_end = index, index + self.seq_len

            r_begin, r_end = s_end + self.lag, s_end + self.pred_len + self.lag

        seq_x = self.data_x[s_begin:s_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        # select y.
        target_y = self.data_y[r_begin:r_end]
        target_mark = self.data_stamp[r_begin:r_end]

        # if self.inverse:
        #     seq_y = np.concatenate(
        #         [self.data_x[r_begin:r_begin + self.label_len], self.data_y[r_begin + self.label_len:r_end]],
        #         axis=0
        #     )
        # return seq_x, seq_y, seq_x_mark, target_y_mark
        return DatasetItem(
            prev_x=seq_x,
            prev_y=None,
            prev_mark=seq_x_mark,
            target_x=None,
            target_y=target_y,
            target_mark=target_mark,
        )

    def __len__(self):
        if self.real_time:
            # round down since index begin from 0!
            assert hasattr(self, "pred_start")
            # return math.floor((len(self.data_x) - self.pred_start) / self.pred_len)
            return math.ceil((len(self.data_x) - self.pred_start) / self.pred_len)
        return len(self.data_x) - self.seq_len - self.pred_len + 1 - self.lag

    def whole_pred_len(self):
        return len(self.data_x) - self.pred_start

    def inverse_transform(self, data):
        if self.features == "MS":
            if isinstance(self.scaler, StandardScaler):
                inversed_data = data[..., -1:] * self.scaler.scale_[-1] + self.scaler.mean_[-1]
            # TODO(rzhao): other inverse transform methods
            elif isinstance(self.scaler, MinMaxScaler):
                inversed_data = (
                        data[..., -1:] * (self.scaler.data_max_[-1] - self.scaler.data_min_[-1])
                        + self.scaler.data_min_[-1]
                )
                pass
            else:
                inversed_data = data
        else:
            inversed_data = self.scaler.inverse_transform(data, copy=True)
        return inversed_data


class Dataset_Gefcom_Reg(Dataset_Gefcom):
    def __init__(
            self, root_path, flag='train', size=None,
            features='S', data_path='ETTh1.csv', target='OT', scale=True,
            inverse=False, timeenc=0, freq='h', cols=None,
            **kwargs
    ):
        # size [seq_len, label_len, pred_len]
        # info
        super().__init__(
            root_path, flag, size, features, data_path, target,
            scale, inverse, timeenc, freq, cols, **kwargs
        )

    def __read_data__(self):

        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        # all_data = df_raw[['ts', 'zone', 'demand', 'drybulb', 'dewpnt', 'holiday_name', 'holiday']]
        df_raw = df_raw[['ts', 'zone', 'demand', 'drybulb', 'dewpnt', ]]
        df_raw.rename(columns={"ts": "date"}, inplace=True)
        # df_raw = df_raw.groupby('zone').get_group(self.zone)
        df_raw = df_raw.loc[df_raw['zone'] == self.zone].drop(columns=['zone']).sort_values(by='date')

        df_raw['date'] = pd.to_datetime(df_raw['date'])
        train_start, train_end = pd.Timestamp(2013, 1, 1), pd.Timestamp(2014, 7, 1)
        valid_end, test_end = pd.Timestamp(2015, 1, 1), pd.Timestamp(2015, 12, 31)
        train_data = df_raw.loc[
            # df_raw['date'] < pd.Timestamp(2016, 7, 1)
            df_raw['date'].apply(lambda x: train_start <= x < train_end)
        ]
        valid_data = df_raw.loc[
            df_raw['date'].apply(lambda x: train_end <= x < valid_end)
        ]
        test_data = df_raw.loc[
            # pd.Timestamp(2016, 10, 1) <= df_raw['date']
            df_raw['date'].apply(lambda x: valid_end <= x < test_end)
        ]
        train_start = sum(df_raw['date'] < train_start)
        valid_start = sum(df_raw['date'] < train_end)
        test_start = sum(df_raw['date'] < valid_end)

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        # cols = list(df_raw.columns);
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]

        num_train, num_valid, num_test = len(train_data), len(valid_data), len(test_data)
        data_split = {
            "train": {
                "start": train_start - int(self.real_time) * (self.seq_len + self.lag),
                "end": train_start + num_train
            },
            "valid": {
                "start": valid_start - self.seq_len - self.lag,
                "end": valid_start + num_valid
            },
            "test": {
                "start": test_start - self.seq_len - self.lag,
                "end": test_start + num_test
            },
        }
        border1, border2 = data_split[self.set_type]['start'], data_split[self.set_type]['end']
        if self.real_time:
            if self.set_type == 'train':
                pred_start = train_start
            elif self.set_type == 'valid':
                pred_start = valid_start
            else:
                pred_start = test_start
            pred_start -= data_split[self.set_type]['start']
            self.set_pred_start(pred_start=pred_start)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        else:
            raise ValueError

        if self.scale:
            # train_data = df_data[border1s[0]:border2s[0]]
            train_data = df_data[data_split['train']['start']:data_split['train']['end']]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
            # TODO(rzhao): inverse_transform the prediction.
        else:
            data = df_data.values

        # df_stamp = df_raw[['date']][border1:border2]
        # df_stamp['date'] = pd.to_datetime(df_stamp.date)
        # data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        else:
            raise ValueError

        self.data_x = data[border1:border2, :-1]
        if self.inverse:
            self.data_y = df_data.values[border1:border2, -1:]
        else:
            self.data_y = data[border1:border2, -1:]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        if self.real_time:
            if index+1 == self.__len__():
                r_begin = len(self.data_x) - self.pred_len
                r_end = len(self.data_x)
            else:
                r_begin = self.pred_start + index * self.pred_len
                r_end = r_begin + self.pred_len
            s_begin = r_begin - self.seq_len - self.lag
            s_end = s_begin + self.seq_len
        else:
            s_begin, s_end = index, index + self.seq_len
            r_begin, r_end = s_end + self.lag, s_end + self.pred_len + self.lag

        prev_x = self.data_x[s_begin:s_end]
        prev_y = self.data_y[s_begin:s_end]
        prev_mark = self.data_stamp[s_begin:s_end]
        # select y.
        target_x = self.data_x[r_begin:r_end]
        target_y = self.data_y[r_begin:r_end]
        target_mark = self.data_stamp[r_begin:r_end]

        # return DatasetItem(
        #     prev_x=prev_x,
        #     prev_y=prev_y,
        #     prev_mark=prev_mark,
        #     target_x=target_x,
        #     target_y=target_y,
        #     target_mark=target_mark,
        # )
        return {
            "prev_x": prev_x,
            "prev_y":  prev_y,
            "prev_mark": prev_mark,
            "target_x": target_x,
            "target_y": target_y,
            "target_mark": target_mark,
        }


# TODO(rzhao).
class Dataset_Gefcom_Reg_Lag(Dataset_Gefcom_Reg):
    def __init__(
            self, root_path, flag='train', size=None, features='S', data_path='ETTh1.csv', target='OT', scale=True,
            inverse=False, timeenc=0, freq='h', cols=None, **kwargs
    ):
        # size [seq_len, label_len, pred_len]
        # info
        super().__init__(
            root_path, flag, size, features, data_path, target, scale, inverse, timeenc, freq, cols, **kwargs
        )

    def __getitem__(self, index):
        if self.real_time:
            if index+1 == self.__len__():
                r_begin = len(self.data_x) - self.pred_len
                r_end = len(self.data_x)
            else:
                r_begin = self.pred_start + index * self.pred_len
                r_end = r_begin + self.pred_len
            s_begin = r_begin - self.seq_len - self.lag
            s_end = s_begin + self.seq_len
        else:
            s_begin, s_end = index, index + self.seq_len
            r_begin, r_end = s_end + self.lag, s_end + self.pred_len + self.lag

        prev_x = self.data_x[s_begin:s_end]
        prev_y = self.data_y[s_begin:s_end]
        prev_mark = self.data_stamp[s_begin:s_end]
        # select y.
        target_x = self.data_x[r_begin:r_end]
        target_y = self.data_y[r_begin:r_end]
        target_mark = self.data_stamp[r_begin:r_end]


        # if self.inverse:
        #     seq_y = np.concatenate(
        #         [self.data_x[r_begin:r_begin + self.label_len], self.data_y[r_begin + self.label_len:r_end]],
        #         axis=0
        #     )
        # return seq_x, seq_y, seq_x_mark, target_x, target_y, target_y_mark

        return {
            "prev_x": prev_x,
            "prev_y":  prev_y,
            "prev_mark": prev_mark,
            "target_x": target_x,
            "target_y": target_y,
            "target_mark": target_mark,
        }
