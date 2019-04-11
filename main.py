import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib.rnn import BasicRNNCell, BasicLSTMCell, GRUCell

from baselines.RNNs import RecurrentModel
from models.dgcgru import DGCGRU
from models.dgcrn import DGCRN
from utils.generator import list_files, batch_data_generator_with_external_data, batch_data_generator, \
    batch_data_generator_for_baselines

frac_train, frac_validation = 0.6, 0.2  # 训练集和验证集所占比例，剩下的是测试集合

batch_size = 5
seq_len = 40
data_dir = 'data/taxi-data/train-data'
graphs_path = 'data/taxi-data/graph-24.npy'
external_data_path = 'data/weathers.h5'

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 指定使用GPU


def get_data_list_by_hour(data_list: list, hour: int):
    new_list = list()
    for item in data_list:
        if hour == int(item[-8:-6]):
            new_list.append(item)
    return new_list


def test_model():
    # graphs = np.load(graphs_path)
    # list_data = list_files(data_dir)
    # external_data: pd.DataFrame = pd.read_hdf(external_data_path)

    # model = DGCRN('dgcrn-bike', num_nodes=graphs.shape[1], batch_size=batch_size, seq_len=seq_len, n_hidden=50,
    #               num_external_features=len(external_data.columns), learning_rate=1e-4)

    # model = DGCGRU('dgcgru-bike-3', num_nodes=graphs.shape[1], batch_size=batch_size, seq_len=seq_len, n_hidden=50,
    #                learning_rate=1e-4)

    # model = DGCGRU('dgcgru-taxi-3', num_nodes=graphs.shape[1], batch_size=batch_size, seq_len=seq_len, n_hidden=50,
    #                learning_rate=1e-3)

    # model = DGCRN('dgcrn-taxi', num_nodes=graphs.shape[1], batch_size=batch_size, seq_len=seq_len, n_hidden=50,
    #               num_external_features=len(external_data.columns), learning_rate=1e-3)
    num_nodes = np.load(graphs_path).shape[1]
    list_data = list_files(data_dir)
    model = RecurrentModel('lstm-taxi', cell=BasicLSTMCell(5 * num_nodes), num_nodes=num_nodes,
                           batch_size=batch_size, seq_len=seq_len, learning_rate=1e-3)

    # 启动session
    config = tf.ConfigProto()  # 定义TensorFlow配置
    config.gpu_options.allow_growth = True  # 配置GPU内存分配方式，按需增长，很关键
    config.gpu_options.per_process_gpu_memory_fraction = 1  # 配置可使用的显存比例

    sess = tf.InteractiveSession(config=config)

    vv = int(len(list_data) * (frac_train + frac_validation))
    generators = dict()
    for hour in range(24):
        tst_list = get_data_list_by_hour(list_data[vv:], hour)
        generators[hour] = batch_data_generator_for_baselines(batch_size, seq_len, tst_list)

    model.restore(sess)
    for hour, generator in generators.items():
        model.predict(sess, generator, results_name='result-{}'.format(hour))


def test_rnns():
    num_nodes = np.load(graphs_path).shape[1]
    list_data = list_files(data_dir)

    model = RecurrentModel('gru-taxi-3', cell=GRUCell(10 * num_nodes), num_nodes=num_nodes,
                           batch_size=batch_size, seq_len=seq_len, learning_rate=1e-3)

    tv, vv = int(len(list_data) * frac_train), int(len(list_data) * (frac_train + frac_validation))

    trn_generator = batch_data_generator_for_baselines(batch_size, seq_len, list_data[:tv])
    val_generator = batch_data_generator_for_baselines(batch_size, seq_len, list_data[tv:vv])
    tst_generator = batch_data_generator_for_baselines(batch_size, seq_len, list_data[vv:])

    # 启动session
    config = tf.ConfigProto()  # 定义TensorFlow配置
    config.gpu_options.allow_growth = True  # 配置GPU内存分配方式，按需增长，很关键
    config.gpu_options.per_process_gpu_memory_fraction = 1  # 配置可使用的显存比例

    sess = tf.InteractiveSession(config=config)

    sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

    model.partial_fit(sess, 20, train_input_generator=trn_generator, validate_input_generator=val_generator)

    model.restore(sess)
    model.predict(sess, tst_generator)


def test_dgcrn():
    graphs = np.load(graphs_path)
    list_data = list_files(data_dir)
    external_data: pd.DataFrame = pd.read_hdf(external_data_path)

    model = DGCRN('dgcrn-bike', num_nodes=graphs.shape[1], batch_size=batch_size, seq_len=seq_len, n_hidden=50,
                  num_external_features=len(external_data.columns), learning_rate=1e-4)

    # Taxi Best Results
    # model = DGCRN('dgcrn-taxi', num_nodes=graphs.shape[1], batch_size=batch_size, seq_len=seq_len, n_hidden=50,
    #               num_external_features=len(external_data.columns), learning_rate=1e-3)

    # Bike Best Results
    # model = DGCRN('dgcrn-bike', num_nodes=graphs.shape[1], batch_size=batch_size, seq_len=seq_len, n_hidden=50,
    #               num_external_features=len(external_data.columns), learning_rate=1e-3)

    tv, vv = int(len(list_data) * frac_train), int(len(list_data) * (frac_train + frac_validation))

    trn_generator = batch_data_generator_with_external_data(batch_size, seq_len, graphs, list_data[:tv], external_data)
    val_generator = batch_data_generator_with_external_data(batch_size, seq_len, graphs, list_data[tv:vv],
                                                            external_data)
    tst_generator = batch_data_generator_with_external_data(batch_size, seq_len, graphs, list_data[vv:], external_data)

    # 启动session
    config = tf.ConfigProto()  # 定义TensorFlow配置
    config.gpu_options.allow_growth = True  # 配置GPU内存分配方式，按需增长，很关键
    config.gpu_options.per_process_gpu_memory_fraction = 1  # 配置可使用的显存比例

    sess = tf.InteractiveSession(config=config)

    sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

    model.partial_fit(sess, 15, train_input_generator=trn_generator, validate_input_generator=val_generator)

    model.predict(sess, tst_generator)


def test_dgcgru():
    graphs = np.load(graphs_path)
    list_data = list_files(data_dir)

    model = DGCGRU('dgcgru-bike-3', num_nodes=graphs.shape[1], batch_size=batch_size, seq_len=seq_len, n_hidden=50,
                   learning_rate=1e-4)

    # Taxi Best Results
    # model = DGCGRU('dgcgru-taxi-3', num_nodes=graphs.shape[1], batch_size=batch_size, seq_len=seq_len, n_hidden=50,
    #                learning_rate=1e-3)

    # Bike Best Results
    # model = DGCGRU('dgcgru-bike-3', num_nodes=graphs.shape[1], batch_size=batch_size, seq_len=seq_len, n_hidden=50,
    #                learning_rate=1e-3)

    tv, vv = int(len(list_data) * frac_train), int(len(list_data) * (frac_train + frac_validation))

    trn_generator = batch_data_generator(batch_size, seq_len, graphs, list_data[:tv])
    val_generator = batch_data_generator(batch_size, seq_len, graphs, list_data[tv:vv])
    tst_generator = batch_data_generator(batch_size, seq_len, graphs, list_data[vv:])

    # 启动session
    config = tf.ConfigProto()  # 定义TensorFlow配置
    config.gpu_options.allow_growth = True  # 配置GPU内存分配方式，按需增长，很关键
    config.gpu_options.per_process_gpu_memory_fraction = 1  # 配置可使用的显存比例

    sess = tf.InteractiveSession(config=config)

    sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

    model.partial_fit(sess, 15, train_input_generator=trn_generator, validate_input_generator=val_generator)

    model.predict(sess, tst_generator)
