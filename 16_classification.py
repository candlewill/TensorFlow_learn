import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

# number 1 to 10 data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)


def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function:
        output = activation_function(Wx_plus_b)
    else:
        output = Wx_plus_b

    return output


def comput_accuarcy(sess, prediction, x, y):
    y_pred = sess.run(prediction, feed_dict={xs: x})
    correct_predict = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuary = tf.reduce_mean(tf.cast(correct_predict, tf.float32))
    result = sess.run(accuary, feed_dict={xs: x, ys: y})
    return result


# define placeholder
xs = tf.placeholder(tf.float32, [None, 784])  # 28*28
ys = tf.placeholder(tf.float32, [None, 10])

# layer
prediction = add_layer(xs, 784, 10, tf.nn.softmax)

# loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    for i in range(10000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
        if i % 50 == 0:
            print(comput_accuarcy(sess, prediction, mnist.test.images, mnist.test.labels))
