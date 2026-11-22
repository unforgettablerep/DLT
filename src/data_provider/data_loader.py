import os
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from src.utils.timefeatures import time_features

warnings.filterwarnings("ignore")

SPLIT_INDEX = {"train": 0, "val": 1, "test": 2}
PEMS_START = {"PEMS03": "2018-09-01", "PEMS04": "2018-01-01", "PEMS07": "2017-05-01", "PEMS08": "2016-07-01"}


def _read_csv(root_path, data_path):
    path = os.path.join(root_path, data_path)
    if not os.path.exists(path):
        raise FileNotFoundError("{} is missing: bash scripts/download_data.sh fetches every dataset".format(path))
    return pd.read_csv(path)


def _stamp_marks(stamps, timeenc, freq, minute_bins):
    # stamps: DatetimeIndex of the split -> (T, n_marks)
    if timeenc == 1:
        return time_features(stamps, freq=freq).transpose(1, 0)
    marks = [stamps.month, stamps.day, stamps.weekday, stamps.hour]
    if minute_bins:
        marks.append(stamps.minute // 15)
    return np.stack(marks, axis=1)


class _WindowDataset(Dataset):
    def __init__(self, args, root_path, flag="train", size=None, features="S", data_path="ETTh1.csv",
                 target="OT", scale=True, timeenc=0, freq="h"):
        self.args = args
        if size is None:
            self.seq_len, self.label_len, self.pred_len = 24 * 4 * 4, 24 * 4, 24 * 4
        else:
            self.seq_len, self.label_len, self.pred_len = size
        assert flag in SPLIT_INDEX
        self.set_type = SPLIT_INDEX[flag]
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.__read_data__()

    def __read_data__(self):
        raise NotImplementedError

    def _scale(self, values, train_end):
        if self.scale:
            self.scaler.fit(values[:train_end])
            return self.scaler.transform(values)
        return values

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        return self.data_x[s_begin:s_end], self.data_y[r_begin:r_end], self.data_stamp[s_begin:s_end], self.data_stamp[r_begin:r_end]

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class _CsvDataset(_WindowDataset):
    minute_bins = False

    def _borders(self, df_raw):
        raise NotImplementedError

    def _frame(self, df_raw):
        return df_raw

    def __read_data__(self):
        df_raw = self._frame(_read_csv(self.root_path, self.data_path))
        border1s, border2s = self._borders(df_raw)
        border1, border2 = border1s[self.set_type], border2s[self.set_type]
        df_data = df_raw[df_raw.columns[1:]] if self.features in ("M", "MS") else df_raw[[self.target]]
        data = self._scale(df_data.values, border2s[0])
        stamps = pd.DatetimeIndex(pd.to_datetime(df_raw["date"].values[border1:border2]))
        self.data_stamp = _stamp_marks(stamps, self.timeenc, self.freq, self.minute_bins)
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]


class Dataset_ETT_hour(_CsvDataset):
    def _borders(self, df_raw):
        month = 30 * 24
        border1s = [0, 12 * month - self.seq_len, 12 * month + 4 * month - self.seq_len]
        border2s = [12 * month, 12 * month + 4 * month, 12 * month + 8 * month]
        return border1s, border2s


class Dataset_ETT_minute(_CsvDataset):
    minute_bins = True

    def _borders(self, df_raw):
        month = 30 * 24 * 4
        border1s = [0, 12 * month - self.seq_len, 12 * month + 4 * month - self.seq_len]
        border2s = [12 * month, 12 * month + 4 * month, 12 * month + 8 * month]
        return border1s, border2s


class Dataset_Custom(_CsvDataset):
    def _frame(self, df_raw):
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove("date")
        return df_raw[["date"] + cols + [self.target]]

    def _borders(self, df_raw):
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        return border1s, border2s


class Dataset_PEMS(_WindowDataset):
    """PEMS03/04/07/08 flow (.npz key "data", feature 0); split 6:2:2, look-back from the preceding split unless --pems_split itransformer."""

    def __read_data__(self):
        data = np.load(os.path.join(self.root_path, self.data_path), allow_pickle=False)["data"]
        if data.ndim == 3:
            data = data[:, :, 0]
        data = pd.DataFrame(data.astype(np.float64)).ffill().bfill().values
        n = len(data)
        border1s = [0, int(0.6 * n) - self.seq_len, int(0.8 * n) - self.seq_len]
        if getattr(self.args, "pems_split", "tslib") == "itransformer":
            border1s = [0, int(0.6 * n), int(0.8 * n)]
        border2s = [int(0.6 * n), int(0.8 * n), n]
        border1, border2 = border1s[self.set_type], border2s[self.set_type]
        if self.features == "S":
            data = data[:, [-1]]
        data = self._scale(data, border2s[0])
        name = os.path.splitext(os.path.basename(self.data_path))[0].upper()
        stamps = pd.date_range(PEMS_START.get(name, "2016-07-01"), periods=n, freq="5min")[border1:border2]
        self.data_stamp = _stamp_marks(stamps, self.timeenc, self.freq, True)
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]


class Dataset_Solar(_WindowDataset):
    """Solar-Energy (solar_AL.txt, 137 plants, 10-minute steps of 2006); split 7:1:2 as the iTransformer / S-Mamba loader."""

    def __read_data__(self):
        data = pd.read_csv(os.path.join(self.root_path, self.data_path), header=None).values.astype(np.float64)
        n = len(data)
        num_train, num_test = int(n * 0.7), int(n * 0.2)
        num_vali = n - num_train - num_test
        border1s = [0, num_train - self.seq_len, n - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, n]
        border1, border2 = border1s[self.set_type], border2s[self.set_type]
        if self.features == "S":
            data = data[:, [-1]]
        data = self._scale(data, border2s[0])
        stamps = pd.date_range("2006-01-01", periods=n, freq="10min")[border1:border2]
        self.data_stamp = _stamp_marks(stamps, self.timeenc, self.freq, True)
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
