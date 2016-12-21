import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

'''
RNN example in TensorFlow

MNIST classification
'''

# data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# hyper-parameters
lr = 0.001
training_iters = 100000
batch_size = 128
display_step = 10

n_inputs = 28  # MNIST data input (img shape: 28*28)，每行作为一个输入
n_steps = 28  # sequence length，一共有28行，因此序列长度为28
n_hidden_units = 128  # hidden layer neurons dim
n_classes = 10

# TF graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# Define weights
weights = {
    # (28, 128)
    "in": tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # (128, 10)
    "out": tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}

# Define biases
biases = {
    # 128
    "in": tf.Variable(tf.constant(0.1, shape=[n_hidden_units])),
    # 10
    "out": tf.Variable(tf.constant(0.1, shape=[n_classes]))
}


# RNN
def RNN(X, weights, biases):
    # hidden layer for input to cell
    # X shape: [128 batch, 28 step, 28 inputs]
    # convert X shape to ==> [128*28, 28 inputs]
    # 特别注意：每一个batch，神经元权值共享
    # 将同一个batch中的图片拼接在一起，以实现权值共享
    X = tf.reshape(X, [-1, n_inputs])  # output dim: [128 batch * 28 steps, 28 inputs]
    X_in = tf.matmul(X, weights["in"]) + biases["in"]  # output dim: [128*28, 128]
    # 还原成多个图片
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])  # output dim: [128 batch, 28 steps, 128 hidden]

    # cell
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    # lstm cell output is divided into two parts (c_state, m_state)
    # c_state: cell state,  memory state
    # m_state: hidden state

    _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    output, states = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=_init_state,
                                       time_major=False)  # time_major控制time_step在哪一个维度，如果在第一个维度，则取值True，第二维度，则取值False

    '''
    The Returns of dynamic_rnn:
    A pair (outputs, state) where:

      outputs: The RNN output `Tensor`.

        If time_major == False (default), this will be a `Tensor` shaped:
          `[batch_size, max_time, cell.output_size]`.

        If time_major == True, this will be a `Tensor` shaped:
          `[max_time, batch_size, cell.output_size]`.

        Note, if `cell.output_size` is a (possibly nested) tuple of integers
        or `TensorShape` objects, then `outputs` will be a tuple having the
        same structure as `cell.output_size`, containing Tensors having shapes
        corresponding to the shape data in `cell.output_size`.

      state: The final state.  If `cell.state_size` is an int, this
        will be shaped `[batch_size, cell.state_size]`.  If it is a
        `TensorShape`, this will be shaped `[batch_size] + cell.state_size`.
        If it is a (possibly nested) tuple of ints or `TensorShape`, this will
        be a tuple having the corresponding shapes.
    '''

    # hidden layer for output as the final results
    # Two methods:
    # Method 1:
    # results = tf.matmul(final_state[1], weights['out']) + biases['out']
    # or Method 2:
    # unpack to list [(batch, outputs)..] * steps

    # Method 1:
    # results = tf.matmul(states[1], weights["out"]) + biases["out"]

    # Method 2:
    output = tf.unpack(tf.transpose(output, [1, 0, 2]))  # states is the last outputs
    results = tf.matmul(output[-1], weights['out']) + biases['out']

    return results


pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    step = 0
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run(train_op, feed_dict={x: batch_xs, y: batch_ys})

        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys}))

        step += 1
