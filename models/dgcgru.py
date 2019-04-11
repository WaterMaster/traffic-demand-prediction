import datetime
import os
from typing import Union, Callable, Generator, Type

import numpy as np
import tensorflow as tf
from tensorflow.contrib.metrics import streaming_pearson_correlation
from tensorflow.python.ops.rnn import dynamic_rnn

from models.dgcgru_cell import DGCGRUCell
from utils.logger import Logger


class DGCGRU(object):
    log_dir = 'logs'
    model_save_dir = os.path.join('data', 'model')

    def __init__(self, model_name: str, learning_rate: float = 1e-3, batch_size: int = 5, n_hidden: int = 5,
                 num_nodes: int = None, seq_len: int = 40, input_dim: int = 2, output_dim: int = 2):
        if num_nodes is None:
            raise ValueError('Please Specific the Number of Nodes.')

        self.logger = Logger(os.path.join(self.log_dir, '{}.log'.format(model_name)), level='debug').logger
        self.model_save_path = os.path.join(self.model_save_dir, model_name, model_name)

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_hidden = n_hidden
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.metrics = dict()

        self._make_graph()
        self._define_metrics()
        self._define_loss()

    def _make_graph(self):
        x = tf.placeholder("float", [self.batch_size, self.seq_len, self.num_nodes, self.num_nodes + self.input_dim],
                           name='input')
        y = tf.placeholder("float", [self.batch_size, self.num_nodes, self.output_dim], name='label')

        self.input = x
        self.label = y

        # 创建DGGRU
        encoding_cell = DGCGRUCell(self.n_hidden, self.num_nodes, self.input_dim)
        outputs, state = dynamic_rnn(encoding_cell, x, dtype=tf.float32)
        # outputs = tf.transpose(outputs, [1, 0, 2])
        pred = tf.contrib.layers.fully_connected(state, self.num_nodes * self.output_dim, activation_fn=None)

        # Define loss and optimizer
        pred = tf.reshape(pred, [self.batch_size, self.num_nodes, self.output_dim], name='predictions')

        self.prediction = pred

    def _define_metrics(self):
        mse = tf.reduce_mean(tf.square(self.prediction - self.label), name='mse')
        mae = tf.reduce_mean(tf.abs(self.prediction - self.label), name='mae')
        pcc = streaming_pearson_correlation(self.prediction, self.label, name='pcc')

        self.metrics['mse'] = mse
        self.metrics['mae'] = mae
        self.metrics['pcc'] = pcc

    def _define_loss(self):
        cost = self.metrics['mse']
        self.loss = self.metrics['mse']

        # global_steps = tf.Variable(0, trainable=False)
        # # Learning rate decay with rate 0.7 every 50 epochs.
        # lr = tf.train.exponential_decay(self.learning_rate, global_steps, decay_steps=550, decay_rate=0.7)
        # step_op = tf.assign_add(global_steps, 1)
        # with tf.control_dependencies([step_op]):
        #     self.optimizer = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(cost)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)

    def restore(self, sess: Union[tf.Session, tf.InteractiveSession]):
        saver = tf.train.Saver()
        saver.restore(sess, self.model_save_path)
        sess.run(tf.local_variables_initializer())

    def partial_fit(self, sess: Union[tf.Session, tf.InteractiveSession], training_epochs: int, display_step: int = 10,
                    train_input_generator: Callable[[], Generator] = None,
                    validate_input_generator: Callable[[], Generator] = None):
        saver = tf.train.Saver()

        epoch, min_loss = 1, float('inf')
        while epoch <= training_epochs:
            losses, total_start = list(), datetime.datetime.now()
            for idx, (batch_x, batch_y) in enumerate(train_input_generator()):
                start_time = datetime.datetime.now()
                # Run optimization op (backprop)
                _, loss = sess.run([self.optimizer, self.loss],
                                   feed_dict={self.input: batch_x, self.label: batch_y})
                if idx % display_step == 0:
                    # Calculate batch loss
                    self.logger.info(
                        "\tEpoch {}, Iter {}: Batch MSE= {:.6f}, Duration= {}s".format(
                            epoch, idx, loss, str((datetime.datetime.now() - start_time).seconds)))
                losses.append(loss)
                del batch_x, batch_y
            self.logger.info(
                "Epoch {}: mean MSE: {:.6f}, total Duration: {}s".format(
                    epoch, np.mean(losses), (datetime.datetime.now() - total_start).seconds))

            # 验证并保存验证损失最小的模型
            losses, total_start = list(), datetime.datetime.now()
            for idx, (test_data, test_label) in enumerate(validate_input_generator()):
                start_time = datetime.datetime.now()
                loss = sess.run([self.loss], feed_dict={self.input: test_data, self.label: test_label})
                if idx % display_step == 0:
                    self.logger.info(
                        "\tValidation Iter {}: MSE: {}, Duration: {}s".format(
                            idx, loss, (datetime.datetime.now() - start_time).seconds))
                losses.append(loss)
                del test_data, test_label
            self.logger.info(
                'Validation mean MSE: {}, total Duration: {}s'.format(np.mean(losses),
                                                                      (datetime.datetime.now() - total_start).seconds))

            if np.mean(losses) < min_loss:
                min_loss = np.mean(losses)
                # 保存训练结果
                saver.save(sess, self.model_save_path)
            epoch += 1

    def predict(self, sess: Union[tf.Session, tf.InteractiveSession], input_generator: Callable[[], Generator],
                results_name='result'):
        predictions, labels, mses, pccs, maes, total_start = list(), list(), list(), list(), list(), datetime.datetime.now()
        for idx, (test_data, test_label) in enumerate(input_generator()):
            prediction, mse, (pcc, _), mae = sess.run(
                [self.prediction, self.metrics['mse'], self.metrics['pcc'], self.metrics['mae']],
                feed_dict={self.input: test_data, self.label: test_label})
            mses.append(mse)
            pccs.append(pcc)
            maes.append(mae)
            predictions.append(prediction)
            labels.append(test_label)
            del test_data, test_label
        np.savez(self.model_save_path + '-{}.npz'.format(results_name), predictions=np.stack(predictions),
                 labels=np.stack(labels))
        self.logger.info(
            'Test mean MSE: {}, Test mean PCC: {}, Test mean MAE: {}, total Duration: {}s'.format(
                np.mean(mses), np.mean(pccs), np.mean(maes), (datetime.datetime.now() - total_start).seconds))
