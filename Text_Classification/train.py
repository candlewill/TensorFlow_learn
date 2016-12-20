# encoding: utf-8
import numpy as np

import tensorflow as tf

'''
本程序作用是使用CNN来分类文本

参考：http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/
'''


class TextCNN(object):
    '''
    用CNN来对文本分类，模型结构：embedding层、卷积层、max-pooling、softmax
    '''

    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters):
        '''
        :param sequence_length: 句子长度，句子有一个固定长度
        :param num_classes: 类别数量
        :param vocab_size: 词汇量，用来定义embedding层的维度 [vocab_size, embedding_size]
        :param embedding_size: 词向量维度
        :param filter_sizes: 一系列filter的大小，使用列表存储，例如 [3, 4, 5]
        :param num_filters: filter对应的数量，每一个大小的filter都有这么多
        '''
        # 定义 place_holder
        self.input_x = tf.place_holder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.place_holder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.place_holder(tf.float32, name="dropout_keep_prob")

        # embedding 层
        with tf.device("/cpu:0"), tf.name_scope(
                "embedding"):  # 目前embedding层的lookup只支持在cpu上运行；使用name_scope可以抽象下面的操作为一个整体，称为embedding，使用TensorBoard可视化时更加直观
            W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, -1.0), name="W")
            self.embedding_chars = tf.nn.embedding_lookup(W, self.input_x)  # 范围3维的tensor [None, 句子长度，词向量维度]
            self.embedding_chars_expanded = tf.expand_dims(self.embedding_chars,
                                                           -1)  # TF的卷积操作，期待4维tensor，因此需要扩维，扩展成[None, sequence_length, embedding_size, 1]

        # 卷积层和max-pooling
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):  # 有多个不同大小的filter，使用循环迭代
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # 卷积层
                filter_shape = [filter_size, embedding_size, 1, num_filters]  # 和句矩阵一样，4维tensor
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(self.embedding_chars_expanded, W, strides=[1, 1, 1, 1], name="conv")
                # Relu 非线性变换
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Max-Pooling
                pool_size = [1, sequence_length - filter_size + 1, 1, 1]
                pooled = tf.nn.max_pool(h, ksize=pool_size, strides=[1, 1, 1, 1], padding="VALID",
                                        name="pool")  # narrow方式，返回的Tensor维度是[batch_size, 1, 1, num_filters]
                pooled_outputs.append(pooled)

        # 合并结果
        num_filters_total = num_filters * len(filter_sizes)  # filter总量
        self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # output
        W=tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name="W")
        b=tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
        self.scores=tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
        self.predictions=tf.argmax(self.scores, name="prediction")

        # loss
        with tf.name_scope("loss"):
            losses=tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss=tf.reduce_mean(losses)

        # accuracy
        with tf.name_scope("accuracy"):
            correct_predicton=tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy=tf.reduce_mean(tf.cast(correct_predicton, tf.float), name="accuracy")