import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

'''
CNN for MNIST classification
'''

# number 1 to 10 data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)


def comput_accuarcy(sess, prediction, x, y):
    y_pred = sess.run(prediction, feed_dict={xs: x, keep_prob: 1.0})
    correct_predict = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuary = tf.reduce_mean(tf.cast(correct_predict, tf.float32))
    result = sess.run(accuary, feed_dict={xs: x, ys: y, keep_prob: 1.})
    return result


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    # strides的定义：[1, x, y, 1]，x横轴方向上的step，y纵轴方向上的step
    # 支持SAME和Valid两种padding方式，same的feature map的维度和输入维度相同，valid不进行填充
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


def max_pool_2x2(x):
    # ksize是pooling大小
    # strides是步长
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

x_image = tf.reshape(xs, [-1, 28, 28, 1])  # 最后一个是通道数量
print(x_image.get_shape)  # [n_samples, 28, 28, 1]

# convolution1
W_conv1 = weight_variable([5, 5, 1, 32])  # 维度含义 [filter_size=5*5, input_chanel=1, output_chanel=32]
b_conv1 = bias_variable([32])
h_conv1 = conv2d(x_image, W_conv1) + b_conv1  # output size: 28*28*32
h_conv1 = tf.nn.relu(h_conv1)  # output size: 14*14*32
# pooling
h_pool1 = max_pool_2x2(h_conv1)

# convolution 2
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = conv2d(h_pool1, W_conv2) + b_conv2  # output dim: 14*14*64
h_conv2 = tf.nn.relu(h_conv2)
h_pool2 = max_pool_2x2(h_conv2)  # output dim: 7*7*64

# convert 3-dim tensor to 1-dim: [n_smaples, 7, 7, 64] >> [n_samples, 7*7*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

# dense layer 1
W_dense1 = weight_variable([7 * 7 * 64, 1024])
b_dense1 = bias_variable([1024])
h_dense1 = tf.matmul(h_pool2_flat, W_dense1) + b_dense1  # output dim: 1024
h_dense1 = tf.nn.relu(h_dense1)
h_dense1 = tf.nn.dropout(h_dense1, keep_prob)

# dense layer 2
W_dense2 = weight_variable([1024, 10])
b_dense2 = bias_variable([10])
h_dense2 = tf.matmul(h_dense1, W_dense2) + b_dense2
prediction = tf.nn.softmax(h_dense2)

# loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
        if i % 50 == 0:
            print(comput_accuarcy(sess, prediction, mnist.test.images, mnist.test.labels))
