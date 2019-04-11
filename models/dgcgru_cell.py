import tensorflow as tf


class DGCGRUCell(tf.nn.rnn_cell.RNNCell):
    """
    Dynamic Graph Convolutional Gated Recurrent Unit.
    """

    def __init__(self, num_units: int, num_nodes: int, input_dim: int, activation=tf.nn.tanh, reuse=True):
        super(DGCGRUCell, self).__init__(_reuse=reuse)
        self._activation = activation
        self._num_nodes = num_nodes
        self._num_units = num_units
        self._input_dim = input_dim

    @property
    def state_size(self):
        return self._num_nodes * self._num_units

    def compute_output_shape(self, input_shape):
        pass

    @property
    def output_size(self):
        output_size = self._num_nodes * self._num_units
        return output_size

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "dcgru_cell"):
            # spliting inputs into signals and graphs
            # signals [batch_size, num_nodes, input_dim]
            # graphs [batch_size, num_nodes, num_nodes]
            signals, graphs = tf.split(inputs, num_or_size_splits=[self._input_dim, self._num_nodes], axis=2)
            with tf.variable_scope("gates"):  # Reset gate and update gate.
                # We start with bias of 1.0 to not reset and not update.
                value = tf.nn.sigmoid(
                    self._gconv(signals, state, graphs, 2 * self._num_units, 1.0))
                r, u = tf.split(value=value, num_or_size_splits=2, axis=1)
                # r, u = sigmoid(r), sigmoid(u)
            with tf.variable_scope("candidate"):
                c = self._gconv(signals, r * state, graphs, self._num_units)
                if self._activation is not None:
                    c = self._activation(c)
            output = new_state = u * state + (1 - u) * c
        return output, new_state

    def _gconv(self, signals, state, graphs, output_size, bias_start=0.0):
        """Graph convolution between input and the graph matrix.
        :param signals: a 2D Tensor or a list of 2D, batch x n, Tensors.
        :param graphs: a 2D Tensor, with shape [batch_size, num_nodes, num_nodes]
        :param state
        :param output_size:
        :return:
        """
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        batch_size = signals.shape[0].value
        signals = tf.reshape(signals, (batch_size, self._num_nodes, -1))
        state = tf.reshape(state, (batch_size, self._num_nodes, -1))
        inputs_and_states = tf.concat([signals, state], axis=2)
        input_size = inputs_and_states.get_shape()[2].value
        dtype = signals.dtype
        result = list()

        weights = tf.get_variable(
            'weights', [input_size, output_size], dtype=dtype, initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable(
            "biases", [output_size], dtype=dtype, initializer=tf.constant_initializer(bias_start, dtype=dtype))

        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            for graph, input_and_state in zip(tf.unstack(graphs), tf.unstack(inputs_and_states)):
                x = input_and_state
                x = tf.reshape(x, shape=[self._num_nodes, input_size])

                theta = tf.get_variable('theta', [], dtype=dtype,
                                        initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
                # TODO theta 的可视化
                x = theta * self.sparse_matmul(self.get_laplace(graph) - self.get_laplace(tf.transpose(graph)), x)
                x = tf.reshape(x, [self._num_nodes, input_size])
                x = tf.nn.bias_add(tf.matmul(x, weights), biases)  # (self._num_nodes, output_size)
                result.append(x)

        result = tf.stack(result)  # [batch_size, num_nodes, output_size]
        # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)
        return tf.reshape(result, [batch_size, self._num_nodes * output_size])

    @staticmethod
    def sparse_matmul(m, x):
        indices = tf.where(tf.not_equal(m, 0))
        values = tf.gather_nd(m, indices)
        shape = tf.shape(m, out_type=tf.int64)
        sparse_m = tf.SparseTensor(indices, values, dense_shape=shape)
        return tf.sparse_tensor_dense_matmul(tf.sparse_reorder(sparse_m), x)

    @staticmethod
    def get_laplace(matrix):
        # return tf.matrix_inverse(tf.diag(tf.reduce_sum(matrix, axis=1))) * matrix
        return tf.nn.l2_normalize(matrix, axis=1)
