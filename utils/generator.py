import os
import random

import numpy as np
import pandas as pd


def list_files(ddir: str):
    names, files = os.listdir(ddir), list()
    names.sort()
    for i in range(0, len(names)):
        path = os.path.join(ddir, names[i])
        if os.path.isdir(path):
            files.extend(list_files(path))
        if os.path.isfile(path):
            files.append(path)
    return files


def batch_data_generator(batch_size: int, seq_len: int, graphs: np.ndarray, list_data: list):
    def generator():
        time_frame = 24 // graphs.shape[0]
        random.shuffle(list_data)
        for i in range(0, len(list_data), batch_size):
            inputs, labels = list(), list()
            for j in range(i, min(i + batch_size, len(list_data))):
                data: pd.DataFrame = pd.read_hdf(list_data[j])
                x, y = data[:seq_len], data[seq_len:]
                xs, ys = list(), list()
                for k in x.index:
                    graph = graphs[k.hour // time_frame]
                    xi = x.loc[k].unstack(level=1).values
                    xs.append(np.concatenate((xi, graph), axis=1))
                inputs.append(np.array(xs))
                for k in y.index:
                    yi = y.loc[k].unstack(level=1).values
                    ys.append(yi)
                labels.append(np.array(ys[0]))
            if len(inputs) == batch_size:
                yield np.array(inputs), np.array(labels)

    return generator


def batch_data_generator_with_external_data(batch_size: int, seq_len: int, graphs: np.ndarray, list_data: list,
                                            external_data: pd.DataFrame):
    def generator():
        time_frame = 24 // graphs.shape[0]
        random.shuffle(list_data)
        for i in range(0, len(list_data), batch_size):
            inputs, labels, e_data = list(), list(), list()
            for j in range(i, min(i + batch_size, len(list_data))):
                data: pd.DataFrame = pd.read_hdf(list_data[j])
                x, y = data[:seq_len], data[seq_len:]
                xs, ys = list(), list()
                for k in x.index:
                    graph = graphs[k.hour // time_frame]
                    xi = x.loc[k].unstack(level=1).values
                    xs.append(np.concatenate((xi, graph), axis=1))
                inputs.append(np.array(xs))
                for k in y.index:
                    yi = y.loc[k].unstack(level=1).values
                    ys.append(yi)
                labels.append(np.array(ys[0]))
                e_data.append(external_data.loc[data.index[-2]].values)
            if len(inputs) == batch_size:
                yield np.array(inputs), np.array(labels), np.array(e_data)

    return generator


def batch_data_generator_for_baselines(batch_size: int, seq_len: int, list_data: list):
    def generator():
        random.shuffle(list_data)
        for i in range(0, len(list_data), batch_size):
            inputs, labels, e_data = list(), list(), list()
            for j in range(i, min(i + batch_size, len(list_data))):
                data: pd.DataFrame = pd.read_hdf(list_data[j])
                x, y = data[:seq_len], data[seq_len:]
                xs, ys = list(), list()
                for k in x.index:
                    xs.append(x.loc[k].unstack(level=1).values)
                inputs.append(np.array(xs))
                for k in y.index:
                    ys.append(y.loc[k].unstack(level=1).values)
                labels.append(np.array(ys[0]))
            if len(inputs) == batch_size:
                yield np.array(inputs), np.array(labels)

    return generator


def batch_data_generator_for_st_gcn(data: np.ndarray, batch_size: int):
    # shape of data [None, 1000, 1000, 2]
    data_size, height, width, _ = data.shape
    recent, period, trend = 1, 24, 24 * 5
    indices = np.arange(trend * 3, data_size)
    random.shuffle(indices)

    def generator():
        recent_batch_data, period_batch_data, trend_batch_data, batch_y = list(), list(), list(), list()
        for idx in indices:
            y = data[idx]
            recent_data = np.transpose(data[[idx - 3 * recent, idx - 2 * recent, idx - 1 * recent]], axes=[1, 2, 0, 3])
            recent_data = np.reshape(recent_data, [height, width, -1])
            period_data = np.transpose(data[[idx - 3 * period, idx - 2 * period, idx - 1 * period]], axes=[1, 2, 0, 3])
            period_data = np.reshape(period_data, [height, width, -1])
            trend_data = np.transpose(data[[idx - 3 * trend, idx - 2 * trend, idx - 1 * trend]], axes=[1, 2, 0, 3])
            trend_data = np.reshape(trend_data, [height, width, -1])
            recent_batch_data.append(recent_data)
            period_batch_data.append(period_data)
            trend_batch_data.append(trend_data)
            batch_y.append(y)
            if len(batch_y) == batch_size:
                yield np.array(recent_batch_data), np.array(period_batch_data), np.array(trend_batch_data), np.array(
                    batch_y)
                recent_batch_data, period_batch_data, trend_batch_data, batch_y = list(), list(), list(), list()

    return generator