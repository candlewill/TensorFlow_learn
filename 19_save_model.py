import tensorflow as tf

'''
TF model save example
'''

"""
with tf.name_scope("xlv"):
    # Save to file
    W = tf.Variable([[1, 2, 3], [3, 4, 5]], dtype=tf.float32, name="weights")
    b = tf.Variable([3, 2, 1], dtype=tf.float32, name="bias")

init = tf.initialize_all_variables()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    save_path = saver.save(sess, "my_net/save_net.ckpt")
    print("Saved to path:", "my_net/save_net.ckpt")

"""
# Restore Variables
# Redefine the same shape and same type for variables
W1 = tf.Variable(tf.zeros(shape=[2, 3]), dtype=tf.float32, name="xlv/weights")
b1 = tf.Variable(tf.zeros(shape=[3]), dtype=tf.float32, name="xlv/bias")

print(W1.name)


# no need init step when loading variables
saver1 = tf.train.Saver()

with tf.Session() as sess:
    saver1.restore(sess, "my_net/save_net.ckpt")
    print("weights: ", sess.run(W1))
    print("bias: ", sess.run(b1))

