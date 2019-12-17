from tensorflow.contrib.layers import fully_connected
import tensorflow as tf
n_steps = 28
n_inputs = 28
n_neurons = 150
n_outputs = 10
learning_rate = 0.001

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])

# 一维输出
y = tf.placeholder(tf.int32, [None])
# 使用最简单的basicRNNcell
basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
# 使用dynamic_rnn ; 原始输出
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)
# 计算和真实的交叉熵
logits = fully_connected(states, n_outputs, activation_fn=None)
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(xentropy)
# 使用AdamOptimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)
# 计算准确率，只有等于y才是对的，其他都错
correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
init = tf.global_variables_initializer()

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./data/")
# 转换到合理的输入shape
X_test = mnist.test.images.reshape((-1, n_steps, n_inputs))
y_test = mnist.test.labels
# run100遍，每次处理150个输入
n_epochs = 100
batch_size = 150
# 开始循环
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            # 读入数据并reshape
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            X_batch = X_batch.reshape((-1, n_steps, n_inputs))
            # X大写，y小写
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
        # 每次打印一下当前信息
        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)