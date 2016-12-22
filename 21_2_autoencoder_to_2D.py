import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt
import numpy as np

''' Auto-Encoder Example
压缩为2维，并在图片中显示出
'''


class auto_encoder(object):
    def __init__(self, n_input, learning_rate, num_hidden_neurons):
        self.input_size = n_input
        self.learning_rate = learning_rate
        self.num_hidden_neurons = [n_input] + num_hidden_neurons  # 输入层数量n_input

        # input
        self.xs = tf.placeholder(tf.float32, [None, self.input_size])

        self.predict(self.xs)

        self.compute_loss()

        self.optimizer()

    def weight_variable(self, shape):
        return tf.Variable(tf.random_normal(shape), name="weight")

    def bias_variable(self, shape):
        return tf.Variable(tf.random_normal(shape), name="bias")

    # 单个神经元
    def _neurons(self, x, in_size, out_size):
        W = self.weight_variable([in_size, out_size])
        b = self.bias_variable([out_size])
        h = tf.matmul(x, W) + b
        h = tf.nn.sigmoid(h)
        return h

    def encoder(self, x, num_neurons):
        for i, num_neuron in enumerate(num_neurons[:-1]):
            num_next_neuron = num_neurons[i + 1]
            if i == 0:
                h = self._neurons(x, num_neuron, num_next_neuron)
            else:
                h = self._neurons(h, num_neuron, num_next_neuron)

        return h

    def decoder(self, x, num_neurons):
        return self.encoder(x, num_neurons[::-1])

    def predict(self, x):
        encoder_op = self.encoder(x, self.num_hidden_neurons)
        decoder_op = self.decoder(encoder_op, self.num_hidden_neurons)
        self.pred = decoder_op
        self.encoder_result = encoder_op  # 以便输出

    # mse loss
    def compute_loss(self):
        # define loss and optimizer
        self.cost = tf.reduce_mean(tf.pow(self.xs - self.pred, 2))

    # optimizer
    def optimizer(self):
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)


def run():
    # hyper-parameters
    # visualize decoder setting
    learning_rate = 0.0001
    training_epochs = 50
    batch_size = 256
    display_step = 1
    examples_to_show = 10

    # Network parameters
    n_input = 784  # MNIST data input (imag shape 28*28)
    n_hidden_1 = 256  # 1st layer num features
    n_hidden_2 = 128  # 2nd layer num features
    n_hidden_3 = 64
    n_hidden_4 = 16
    n_hidden_5 = 2

    num_hidden_neurons = [n_hidden_1, n_hidden_2, n_hidden_3, n_hidden_4, n_hidden_5]

    mnist = input_data.read_data_sets("/tmp/data", one_hot=False)

    model = auto_encoder(n_input, learning_rate, num_hidden_neurons)

    # Launch the graph
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        total_batches = int(mnist.train.num_examples / batch_size)

        # Training
        for epoch in range(training_epochs):
            # Loop over all batches
            for i in range(total_batches):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                _, cost = sess.run([model.optimizer, model.cost], feed_dict={model.xs: batch_xs})

            # display logs per epoch step
            if epoch % display_step == 0:
                print("epoch: %04d, cost: %.9f" % (epoch + 1, cost))

        print("Optimization Finished!")

        # 观察编码后的结果，既，解码之前的结果，压缩成2维的结果显示出来
        encoder_result = sess.run(model.encoder_result, feed_dict={model.xs: mnist.test.images})  # shape: (10000, 2)
        print("The type of encoder_result is: %s\nIts shape is: %s" % (type(encoder_result), encoder_result.shape))
        # draw encoder result
        plt.scatter(encoder_result[:, 0], encoder_result[:, 1], c=mnist.test.labels)
        plt.show()

        # Applying encoder and decoder to test set
        encode_decode = sess.run(model.pred, feed_dict={
            model.xs: mnist.test.images[:examples_to_show]})  # only show examples_to_show images

        # compare original images with their reconstructions
        f, a = plt.subplots(2, 10, figsize=(10, 2))
        for i in range(examples_to_show):
            a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
            a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
        plt.show()


if __name__ == '__main__':
    run()
