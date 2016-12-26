import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

'''
Batch normalization for NN

Build two neural networks:
1. Without batch normalization
2. With batch normalization

Run tests on these two networks
'''


# Make up input data
def get_data(draw=False):
    x_data = np.linspace(-7, 10, 2500)[:, np.newaxis]
    np.random.shuffle(x_data)
    noise = np.random.normal(0, 8, x_data.shape)  # mean=0, stddev=8
    y_data = np.square(x_data) + noise - 5
    if draw:
        # plot input data
        plt.scatter(x_data, y_data)
        plt.show()
    return x_data, y_data


# fix seed for reproduction
def fix_seed(seed):
    np.random.seed(seed)
    tf.set_random_seed(seed)


def plot_hist(inputs, inputs_norm):
    # plot histogram for the inputs of every layer
    for j, all_inputs in enumerate([inputs, inputs_norm]):
        for i, input in enumerate(all_inputs):
            plt.subplot(2, len(all_inputs), j * len(all_inputs) + (i + 1))
            plt.cla()
            if i == 0:
                the_range = (-7, 10)
            else:
                the_range = (-1, 1)
            input = np.array(input[0])
            plt.hist(input.ravel(), bins=15, range=the_range, color='#FF5733')
            plt.yticks(())
            if j == 1:
                plt.xticks(the_range)
            else:
                plt.xticks(())
            ax = plt.gca()
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
        plt.title("%s normalizing" % ("Without" if j == 0 else "With"))
    plt.draw()
    plt.pause(0.01)


class NN(object):
    def __init__(self, n_layers, n_hidden_units, activation, norm=False):
        self.norm = norm
        print("The value of self.norm is: %s" % self.norm)
        self.n_layers = n_layers
        self.n_hidden_units = n_hidden_units
        self.activation = activation

        with tf.name_scope("input"):
            self.input()

        with tf.name_scope("net"):
            self.build_net()

        with tf.name_scope("train"):
            self.train()

    def train(self):
        cost = tf.reduce_mean(tf.reduce_sum(tf.square(self.ys - self.prediction), reduction_indices=[1]))
        train_op = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
        self.cost, self.train_op = cost, train_op

    def build_net(self):
        # record inputs for every layer
        layers_inputs = [self.xs] if self.norm is False else [self.xs_norm]

        # build hidden layer
        for l_n in range(self.n_layers):
            layers_input = layers_inputs[l_n]
            in_size = layers_input.get_shape()[1].value
            output = self._add_hidden_layer(layers_input,  # input
                                            in_size,  # input size
                                            self.n_hidden_units,  # output size
                                            self.activation  # activation function
                                            )
            layers_inputs.append(output)  # add output for next run

        # build output layer
        prediction = self._add_hidden_layer(layers_inputs[-1], self.n_hidden_units, 1, activation_function=None)
        self.prediction = prediction
        self.layer_inputs = layers_inputs

    def _add_hidden_layer(self, x, in_size, out_size, activation_function=None):
        W = tf.Variable(tf.random_normal([in_size, out_size], mean=0., stddev=1.))
        b = tf.Variable(tf.constant(0.1, shape=[out_size]))

        # fully connected layer
        wx_b = tf.matmul(x, W) + b  # output dim: [n_x, out_size]

        # normalize product of fully connected
        # normalize `wx_b` before input into activation function
        if self.norm:
            wx_b = self.normalization(wx_b, out_size)

        if activation_function is not None:
            h = activation_function(wx_b)
        else:
            h = wx_b
        return h

    def input(self):
        self.xs = tf.placeholder(tf.float32, [None, 1])
        self.ys = tf.placeholder(tf.float32, [None, 1])
        if self.norm:
            self.xs_norm = self.normalization(self.xs, 1)
            # self.xs = xs_norm

        print("The shape of inputs:\nxs: %s, ys: %s" % (self.xs.get_shape(), self.ys.get_shape()))

    def normalization(self, wx_b, out_size):
        # Batch normalization
        fc_mean, fc_var = tf.nn.moments(wx_b, axes=[0])  # Calculate the mean and variance of `wx_b`.

        scale = tf.Variable(tf.ones([out_size]))
        shift = tf.Variable(tf.zeros([out_size]))
        epsilon = 0.001

        # apply moving average for mean and var when train on batch
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_appy_op = ema.apply([fc_mean, fc_var])
            with tf.control_dependencies([ema_appy_op]):
                return tf.identity(fc_mean), tf.identity(fc_var)

        mean, var = mean_var_with_update()

        wx_b = tf.nn.batch_normalization(wx_b, mean, var, shift, scale, epsilon)
        # similar with this two steps:
        # Wx_plus_b = (Wx_plus_b - fc_mean) / tf.sqrt(fc_var + 0.001)
        # Wx_plus_b = Wx_plus_b * scale + shift
        return wx_b


def run():
    # hyper-parameters
    seed = 1
    n_layers = 7
    n_hidden_units = 30
    activation = tf.nn.tanh

    fix_seed(seed)

    x_data, y_data = get_data(True)
    print("The shape of x_data is: %s\ny_data: %s" % (x_data.shape, y_data.shape))
    model = NN(n_layers, n_hidden_units, activation, norm=False)
    normalized_model = NN(n_layers, n_hidden_units, activation, norm=True)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # record cost
        cost_hist = []
        cost_hist_norm = []
        record_step = 5

        plt.ion()
        plt.figure(figsize=(7, 3))

        for i in range(250):
            if i % 50 == 0:
                # plot histogram
                all_inputs = sess.run([model.layer_inputs], feed_dict={model.xs: x_data, model.ys: y_data})
                all_inputs_norm = sess.run([normalized_model.layer_inputs],
                                           feed_dict={normalized_model.xs: x_data, normalized_model.ys: y_data})
                plot_hist(all_inputs, all_inputs_norm)

            # train on batch
            x_batch, y_batch = x_data[i * 10:i * 10 + 10], y_data[i * 10:i * 10 + 10]

            sess.run([model.train_op],
                     feed_dict={model.xs: x_batch, model.ys: y_batch})
            sess.run([normalized_model.train_op],
                     feed_dict={normalized_model.xs: x_batch,
                                normalized_model.ys: y_batch})

            # record cost
            if i % record_step == 0:
                cost_hist.append(sess.run(model.cost, feed_dict={model.xs: x_data, model.ys: y_data}))
                cost_hist_norm.append((sess.run(normalized_model.cost,
                                                feed_dict={normalized_model.xs: x_data, normalized_model.ys: y_data})))

        # show cost
        plt.ioff()
        plt.figure()
        plt.plot(np.arange(len(cost_hist)) * record_step, np.array(cost_hist), label="no BN")
        plt.plot(np.arange(len(cost_hist_norm)) * record_step, np.array(cost_hist_norm), label='BN')  # norm
        plt.legend()
        plt.show()

        print(cost_hist)
        print(cost_hist_norm)


if __name__ == '__main__':
    run()
