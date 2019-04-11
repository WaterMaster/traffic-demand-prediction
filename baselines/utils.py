import numpy as np
import pandas as pd


def load_data_for_baseline(pickup_path, dropoff_path, _seq_len, _horizon=1, freq='1H'):
    _pickup_signal_sequence: pd.DataFrame = pd.read_hdf(pickup_path)
    _dropoff_signal_sequence: pd.DataFrame = pd.read_hdf(dropoff_path)

    xs, ys, data_len = list(), list(), _seq_len
    for now in _pickup_signal_sequence.index:
        x1 = _pickup_signal_sequence[now: now + (_seq_len - 1) * pd.Timedelta(freq)].values
        y1 = _pickup_signal_sequence[
             now + _seq_len * pd.Timedelta(freq): now + (_seq_len + _horizon - 1) * pd.Timedelta(freq)].values
        x2 = _dropoff_signal_sequence[now: now + (_seq_len - 1) * pd.Timedelta(freq)].values
        y2 = _dropoff_signal_sequence[
             now + _seq_len * pd.Timedelta(freq): now + (_seq_len + _horizon - 1) * pd.Timedelta(freq)].values
        x = np.concatenate((x1, x2), axis=1)
        y = np.concatenate((y1, y2), axis=1)
        if len(x) == _seq_len and len(y) == _horizon:
            xs.append(x)
            ys.append(y)

    data = list(zip(xs, ys))
    return data
