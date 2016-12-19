# encoding: utf-8
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

'''
本程序是Stanford CS224d 课程中，关于TensorFlow的练习题

练习使用TensorFlow进行线性回归
'''


def main():
    # 定义输入数据
    x_data = np.arange(100, step=0.1)
    y_data = x_data + 20 * np.sin(x_data / 10)

    # 画图
    # plt.scatter(x_data, y_data)
    # plt.show()

    # 定义样本数量和batch大小
    n_samples = 1000
    batch_size = 100

    # 维度reshape
    x_data = np.reshape(x_data, (n_samples, 1))
    y_data = np.reshape(y_data, (n_samples, 1))

    # 定义placeholder
    x = tf.placeholder(tf.float32, shape=(batch_size, 1))  # 类型和大小
    y = tf.placeholder(tf.float32, shape=(batch_size, 1))

    # 定义权值
    with tf.variable_scope("linear-regression"):
        W = tf.get_variable("weights", (1, 1), initializer=tf.random_normal_initializer())
        b = tf.get_variable("bias", (1,), initializer=tf.constant_initializer(0.0))

        y_pred = tf.matmul(x, W) + b

        loss = tf.reduce_sum((y - y_pred) ** 2) / n_samples

    # 梯度下降
    opt_operation = tf.train.AdamOptimizer().minimize(loss)

    with tf.Session() as sess:
        # 先初始化所有变量
        sess.run(tf.initialize_all_variables())
        # 梯度下降迭代500次
        # minibatch随机选择
        indices = np.random.choice(n_samples, batch_size)
        x_batch, y_batch = x_data[indices], y_data[indices]
        # 开始梯度下降
        _, loss = sess.run([opt_operation, loss], feed_dict={x: x_batch, y: y_batch})
        print(sess.run(W, b))


if __name__ == '__main__':
    main()
