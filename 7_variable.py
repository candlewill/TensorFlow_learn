import tensorflow as tf

state = tf.Variable(0, name="counter")

# print(state.name)

one = tf.constant(1)

new_value = tf.add(state, one)
update = tf.assign(state, new_value)
init = tf.initialize_all_variables()  # 有变量时必须要有

with tf.Session() as sess:
    sess.run(init)
    for _ in range(30):
        sess.run(update)
        print(sess.run(state))
