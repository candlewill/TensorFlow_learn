import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt
import numpy as np

''' Auto-Encoder Example '''


class auto_encoder(object):
    def __init__(self, n_input, learning_rate, n_hidden_1, n_hidden_2):
        self.input_size = n_input
        self.learning_rate = learning_rate
        self.n_hidden_1 = n_hidden_1
        self.n_hidden_2 = n_hidden_2

        # input
        self.xs = tf.placeholder(tf.float32, [None, self.input_size])

        self.predict(self.xs)

        self.compute_loss()

        self.optimizer()

    def weight_variable(self, shape):
        return tf.Variable(tf.random_normal(shape), name="weight")

    def bias_variable(self, shape):
        return tf.Variable(tf.random_normal(shape), name="bias")

    def encoder(self, x):
        '''Two layers NN
        input_size ==> n_hidden_1 ==> n_hidden_2
        '''
        # layer 1
        W1 = self.weight_variable([self.input_size, self.n_hidden_1])
        b1 = self.bias_variable([self.n_hidden_1])
        layer_1 = tf.matmul(x, W1) + b1
        layer_1 = tf.nn.sigmoid(layer_1)

        # layer 2
        W2 = self.weight_variable([self.n_hidden_1, self.n_hidden_2])
        b2 = self.bias_variable([self.n_hidden_2])
        layer_2 = tf.matmul(layer_1, W2) + b2
        layer_2 = tf.nn.sigmoid(layer_2)
        return layer_2

    def decoder(self, x):
        '''Two layers NN
        n_hidden_2 ==> n_hidden_1 ==> input_size
        '''
        # layer 1
        W1 = self.weight_variable([self.n_hidden_2, self.n_hidden_1])
        b1 = self.bias_variable([self.n_hidden_1])
        layer_1 = tf.matmul(x, W1) + b1
        layer_1 = tf.nn.sigmoid(layer_1)

        # layer 2
        W2 = self.weight_variable([self.n_hidden_1, self.input_size])
        b2 = self.bias_variable([self.input_size])
        layer_2 = tf.matmul(layer_1, W2) + b2
        layer_2 = tf.nn.sigmoid(layer_2)
        return layer_2

    def predict(self, x):
        encoder_op = self.encoder(x)
        decoder_op = self.decoder(encoder_op)
        self.pred = decoder_op

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
    learning_rate = 0.01
    training_epochs = 20
    batch_size = 256
    display_step = 1
    examples_to_show = 10

    # Network parameters
    n_input = 784  # MNIST data input (imag shape 28*28)
    n_hidden_1 = 256  # 1st layer num features
    n_hidden_2 = 126  # 2nd layer num features

    mnist = input_data.read_data_sets("/tmp/data", one_hot=False)

    model = auto_encoder(n_input, learning_rate, n_hidden_1, n_hidden_2)

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

        # Applying encoder and decoder to test set
        encode_decode = sess.run(model.pred, feed_dict={model.xs: mnist.test.images[:examples_to_show]})
        # compare original images with their reconstructions
        f, a = plt.subplots(2, 10, figsize=(10, 2))
        for i in range(examples_to_show):
            a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
            a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
        plt.show()


if __name__ == '__main__':
    run()
