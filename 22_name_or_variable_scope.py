import tensorflow as tf

'''
本例展示reuse_variable需要设置成True的一个例子

如果RNN中，Train RNN 和 Test RNN 结构不同的情况下，可以使用reuse_variable实现

RNN的Train和Test结构不同，有许多应用场景，例如，训练时固定n_timesteps为N，但是在测试时，其实，只要test sample 长度小于N，就可以通过共享变量（参数相同），来解码
而不用只能支持，长度为N的序列
'''


class TrainConfig:
    batch_size = 20
    n_timesteps = 20
    input_size = 10
    output_size = 2
    cell_size = 11
    learning_rate = 0.01


class TestConfig(TrainConfig):
    n_timesteps = 1  # 其它参数相同，这个参数不同


class RNN(object):
    def __init__(self, config):
        self.batch_size = config.batch_size
        self.n_timesteps = config.n_timesteps
        self.input_size = config.input_size
        self.output_size = config.output_size
        self.cell_size = config.cell_size
        self.learning_rate = config.learning_rate

        self.build_RNN()

    def build_RNN(self):
        with tf.name_scope("input"):
            self.input()

        with tf.name_scope("RNN"):
            with tf.variable_scope("input_hidden"):
                self.input_hidden()
            with tf.variable_scope("LSTM_cell"):
                self.cell()
            with tf.variable_scope("output_hidden"):
                self.output_hidden()
            with tf.name_scope("cost"):
                self.compute_cost()
            with tf.name_scope("train"):
                self.train()

    def train(self):
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

    def compute_cost(self):
        # reshape pred from [batch*time_step, output_size] to [batch, time_step, output_size]
        pred = tf.reshape(self.pred, [self.batch_size, self.n_timesteps, self.output_size])
        mse = self.mse(pred, self.ys)
        self.cost = mse
        self.cost_ave_time = self.cost / self.n_timesteps

    def mse(self, pred, target):
        mse = tf.square(tf.sub(pred, target))
        mse_ave_across_batch = tf.reduce_mean(mse, 0)
        mse_sum_across_time = tf.reduce_sum(mse_ave_across_batch, 0)
        return mse_sum_across_time

    def output_hidden(self):
        # cell_outputs_reshaped to (BATCH*TIME_STEP, CELL_SIZE)
        reshaped_cell_outputs = tf.reshape(tf.concat(1, self.cell_outputs), [-1, self.cell_size])
        W = self.weight_variable([self.cell_size, self.output_size])
        b = self.bias_variable([self.output_size])
        h = tf.matmul(reshaped_cell_outputs, W) + b
        self.pred = tf.nn.relu(h)  # output dim: [batch*time_step, output_size]

    def cell(self):
        # cell = tf.nn.rnn_cell.BasicLSTMCell(self.cell_size)
        cell = tf.nn.rnn_cell.BasicRNNCell(self.cell_size)
        _init_state = cell.zero_state(self.batch_size, dtype=tf.float32)

        cell_outputs = []

        for t in range(self.n_timesteps):
            if t > 0:
                tf.get_variable_scope().reuse_variables()
            cell_output, cell_state = cell(self.layer_in_y[:, t, :], _init_state if t == 0 else cell_state)
            cell_outputs.append(cell_output)

        self.cell_final_state = cell_state
        self.cell_outputs = cell_outputs

    def input_hidden(self):
        layer_in_x = tf.reshape(self.xs, [-1, self.input_size],
                                name="3D_2_2D")  # output dim: [batch_size*n_timesteps, input_size]
        W = self.weight_variable([self.input_size, self.cell_size])
        b = self.bias_variable([self.cell_size])
        print("W name: %s, b name: %s" % (W.name, b.name))
        if debug:
            print(
                "shape:\nlayer_in_x shape: %s, w shape: %s, b shape: %s" % (
                    layer_in_x.get_shape(), W.get_shape(), b.get_shape()))
        with tf.name_scope("Wx_puls_b"):
            h = tf.matmul(layer_in_x, W) + b
        # reshape back
        self.layer_in_y = tf.reshape(h, [-1, self.n_timesteps, self.cell_size], name="2D_2_3D")

        if debug:
            print("The shape of layer_in_y is: %s" % (self.layer_in_y.get_shape()))

    def input(self):
        self.xs = tf.placeholder(tf.float32, [self.batch_size, self.n_timesteps, self.input_size], name="xs")
        self.ys = tf.placeholder(tf.float32, [self.batch_size, self.n_timesteps, self.output_size], name="ys")

    def weight_variable(self, shape, name="weight"):
        initializer = tf.random_normal_initializer(mean=0., stddev=0.5)
        return tf.get_variable(shape=shape, name=name, initializer=initializer)

    def bias_variable(self, shape, name="bias"):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(shape=shape, name=name, initializer=initializer)


def run():
    train_config = TrainConfig()
    test_config = TestConfig()

    '''
    # 错误方法1：错误的调用RNN的方法
    with tf.variable_scope("train_rnn"):
        train_rnn1 = RNN(train_config)
    with tf.variable_scope("test_rnn"):
        test_rnn1 = RNN(test_config)

    # 错误原因：权值没有实现共享，没有放在同一个variable_scope中

    # 错误方法2：原因是，没有打开reuse=True开关，重用变量时将会报错
    train_rnn1 = RNN(train_config)
    test_rnn1 = RNN(test_config)

    '''

    # 正确的调用RNN的方法
    with tf.variable_scope("rnn") as scope:
        train_rnn2 = RNN(train_config)
        scope.reuse_variables()  # 注意顺序，变量共享，需要在同一个variable_scope中，并且，设置reuse_variable
        test_rnn2 = RNN(test_config)


if __name__ == '__main__':
    debug = True
    run()
