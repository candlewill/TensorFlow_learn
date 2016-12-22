import tensorflow as tf
import numpy as np

from matplotlib import pyplot as plt

'''
回归问题

用一条曲线来预测另一条曲线的走势

问题类似于：
类似于，用两支平行的笔，画了两条曲线，有一条曲线部分抹去，请用另一条曲线来，预测被抹去的部分
'''


# 生成模拟数据
def get_batch(batch_size, n_timesteps):
    global batch_start
    # xs shape: (50 batch, 20 steps)
    xs = np.arange(batch_start, batch_start + batch_size * n_timesteps).reshape(((batch_size, n_timesteps))) / (
        10 * np.pi)  # output dim: (batch_size, n_timestep)
    seq = np.sin(xs)
    res = np.cos(xs)
    batch_start += n_timesteps
    # draw, just show the first batch
    '''
    plt.plot(xs[0, :], res[0, :], "r", xs[0, :], seq[0, :], "b--")
    plt.show()
    '''

    return [seq[:, :, np.newaxis], res[:, :, np.newaxis], xs]


# Model
class LSTMRNN(object):
    def __init__(self, n_timestep, input_size, output_size, cell_size, batch_size, lr):
        self.n_timestep = n_timestep
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size

        # inputs
        with tf.name_scope("inputs"):
            self.xs = tf.placeholder(tf.float32, [None, self.n_timestep, self.input_size], name="xs")
            self.ys = tf.placeholder(tf.float32, [None, self.n_timestep, self.output_size], name="ys")

        # hidden layer 1: in_hidden
        # 需要进行变量共享，因此使用，variable_scope
        with tf.variable_scope("in_hidden"):
            self.add_input_layer()

        # LSTM cell
        # 需要进行变量共享，因此使用，variable_scope
        with tf.variable_scope("LSTM_cell"):
            self.add_cell()

        # hidden layer 2: out_hidden
        # 需要进行变量共享，因此使用，variable_scope
        with tf.variable_scope("out_hidden"):
            self.add_output_layer()

        # loss
        with tf.name_scope("loss"):
            self.compute_cost()

        # train
        with tf.name_scope("train"):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.cost)

    def add_input_layer(self):
        layer_in_x = tf.reshape(self.xs, [-1, self.input_size],
                                name="3D_2_2D")  # convert (batch_size, n_timesteps, input_size) to (batch_size*n_timesteps, input_size)
        W = self._weight_variable([self.input_size, self.cell_size])
        b = self._bias_variable([self.cell_size])

        with tf.name_scope("Wx_plus_b"):
            layer_in_y = tf.matmul(layer_in_x, W) + b  # output dim: (batch_size*n_timesteps, cell_size)

        # reshape back: ==> (batch_size, n_timesteps, cell_size)
        self.layer_in_y = tf.reshape(layer_in_y, [-1, self.n_timestep, self.cell_size], name="2D_2_3D")

    def add_cell(self):
        # cell
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)
        _init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
        self.cell_outputs, self.final_state = tf.nn.dynamic_rnn(lstm_cell, self.layer_in_y, initial_state=_init_state,
                                                                time_major=False)  # time_major控制time_step在哪一个维度，如果在第一个维度，则取值True，第二维度，则取值False
        ''' 注，cell_outputs shape: [batch_size, max_time, cell_size] '''

        self.init_state = _init_state  # 之后会使用上一个batch的final state作为initial state

    def add_output_layer(self):
        # reshape cell_outputs to [batch_size*max_time, cell_size]
        layer_out_x = tf.reshape(self.cell_outputs, [-1, self.cell_size], name="3D_2_2D")
        '''这里可能有问题，不应该和输入权值共享，应该加上name
        没有问题：因为这个函数包在了variable_scope里面，会在前面自动加上前缀
        '''
        W = self._weight_variable([self.cell_size, self.output_size])
        b = self._bias_variable([self.output_size])
        with tf.name_scope("Wx_plus_b"):
            self.pred = tf.matmul(layer_out_x,
                                  W) + b  # output shape: [batch_size*max_time, output_size], output_size=10

    def compute_cost(self):
        # flatten
        pred = tf.reshape(self.pred, [-1], name="reshape_pred")
        target = tf.reshape(self.ys, [-1], name="reshape_target")
        weights = tf.ones([self.batch_size * self.n_timestep], dtype=tf.float32)
        print("shape: \npred: %s, target: %s, weights:%s" % (pred.get_shape(), target.get_shape(), weights.get_shape()))
        '''
        sequence_loss_by_example 参数说明：

        logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
        targets: List of 1D batch-sized int32 Tensors of the same length as logits.
        weights: List of 1D batch-sized float-Tensors of the same length as logits.
        '''
        losses = tf.nn.seq2seq.sequence_loss_by_example(logits=[pred], targets=[target], weights=[weights],
                                                        softmax_loss_function=self.mse, name="losses")
        with tf.name_scope("average_cost"):
            self.cost = tf.div(tf.reduce_sum(losses, name="losses_sum"), self.batch_size, name="average_cost")
            tf.scalar_summary("cost", self.cost)

    def mse(self, y_pred, y_target):
        return tf.square(tf.sub(y_pred, y_target))  # sub: Returns x - y element-wise.

    def _weight_variable(self, shape, name="weights"):
        initializer = tf.random_normal_initializer(mean=0, stddev=1.0)
        # 共享 weights 变量
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    def _bias_variable(self, shape, name="biases"):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(shape=shape, initializer=initializer, name=name)


def run():
    # hyper-parameters
    n_timesteps = 20  # sequence length, namely, n_timestep
    batch_size = 50
    input_size = 1
    output_size = 1
    cell_size = 10
    lr = 0.006

    model = LSTMRNN(n_timesteps, input_size, output_size, cell_size, batch_size, lr)
    with tf.Session() as sess:
        merged = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter("RNN_Regression_log", sess.graph)
        # $ tensorboard --logdir = "RNN_Regression_log"

        # tf.initialize_all_variables() no long valid from
        # 2017-03-02 if using tensorflow >= 0.12
        sess.run(tf.global_variables_initializer())

        # draw
        plt.ion()
        plt.show()

        final_state = None
        for i in range(200):
            # 准备数据
            seq, res, xs = get_batch(batch_size, n_timesteps)
            if i == 0:  # 第一个batch，使用正常的方式训练
                feed_dict = {model.xs: seq, model.ys: res}  # create initial state
            else:  # 之后的batch，使用truncated bp训练，即，上一个batch的final state作为本次batch的initial state
                feed_dict = {model.xs: seq, model.ys: res, model.init_state: final_state}

            # 喂入模型，开始训练
            _, cost, final_state, pred = sess.run([model.train_op, model.cost, model.final_state, model.pred],
                                                  feed_dict=feed_dict)

            # draw

            plt.plot(xs[0, :], res[0].flatten(), 'r', xs[0, :], pred.flatten()[:n_timesteps], 'b--')
            plt.ylim((-1.2, 1.2))
            plt.draw()
            plt.pause(0.3)

            if i % 20 == 0:
                # show cost value
                print("cost: %s, i=%s" % (cost, i))

                # add to summary
                result = sess.run(merged, feed_dict)
                writer.add_summary(result, i)


if __name__ == '__main__':
    # global variables
    batch_start = 0

    run()
