# TensorFlow中LSTM具体实现
import tensorflow as tf
import numpy as np
from input_data import readFile, data_reshape, data_reshape_test, PEOPEL_NUM

n_steps = 40  # X的数量
n_inputs = 8064  # 一个X有n_inputs个数
n_neurons = 128  # RNN神经元数目
n_outputs = 2  # 输出
n_layers = 4  # n_layer层神经元
BATCH_SIZE_ALL = PEOPEL_NUM*40//4*3
BATCH_SIZE = 10
n_epochs = 1000
learning_rate_base = 0.001

signal_data, signal_labels = readFile('F:/情感计算/数据集/DEAP/')
signal_re, labels_re = data_reshape(signal_data, signal_labels)
signal_test_re, labels_test_re = data_reshape_test(signal_data, signal_labels)

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])  # 初始化x
y = tf.placeholder(tf.int32, [None])  # 初始化y

lstm_cells = [tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons)
              for layer in range(n_layers)]  # 生成n_layers层，每层包括n_neurons个神经元的神经元列表
multi_cell = tf.contrib.rnn.MultiRNNCell(lstm_cells)    # 根据神经元列表 构建多层循环神经网络
outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)
# outputs(tensor):[ batch_size, max_time, cell.output_size ]
# states:state是一个tensor。state是最终的状态，也就是序列中最后一个cell输出的状态。一般情况下state的形状为 [batch_size,
# cell.output_size ]，但当输入的cell为BasicLSTMCell时，state的形状为[2，batch_size, cell.output_size ]，其中2也对应着
# LSTM中的cell state和hidden state
top_layer_h_state = states[-1][0]+states[-1][1]
print("------------可爱的分割线------------")
print(outputs)
print("------------可爱的分割线------------")
print(states)
print("------------可爱的分割线------------")
print(np.array(states).shape)
print("------------可爱的分割线------------")
print(np.array(states[-1][1]).shape)
dense1 = tf.layers.dense(top_layer_h_state, 1024, kernel_regularizer=tf.contrib.layers.l1_regularizer(0.003),
                         activation=tf.nn.tanh, use_bias=True)  # 增添了一个全连接层 l1正则化
# dense1 = tf.nn.dropout(dense1, 0.5)
dense2 = tf.layers.dense(dense1, 512, kernel_regularizer=tf.contrib.layers.l1_regularizer(0.003),
                         activation=tf.nn.relu)  # 增添了一个全连接层 l1正则化
logits = tf.layers.dense(dense2, n_outputs,
                         name="softmax")  # 增添了一个全连接层 l1正则化
logits = tf.nn.dropout(logits, 0.5)
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
# 计算logits 和 labels 之间的稀疏softmax 交叉熵
loss_mse = tf.reduce_mean(xentropy, name="loss")
learning_rate = tf.train.exponential_decay(  # 指数衰减学习率
    learning_rate_base,
    n_epochs,
    BATCH_SIZE_ALL/BATCH_SIZE,
    0.97,
    staircase=True
)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss_mse)
correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

# 转换到合理的输入shape
X_test = signal_test_re
y_test = np.array(labels_test_re).reshape(len(signal_test_re))
# run n_epochs遍，每次处理BATCH_SIZE个输入
# 开始循环
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        start = (epoch * BATCH_SIZE) % BATCH_SIZE_ALL
        end = start + BATCH_SIZE
        # 读入数据并reshape
        X_batch = np.array(signal_re[start:end]).reshape(BATCH_SIZE, 40, 8064)
        # print(X_batch)  # 测试用
        y_batch = np.array(labels_re[start:end]).reshape(BATCH_SIZE)
        X_batch = X_batch.reshape((-1, n_steps, n_inputs))  # -1为未指定行数
        # y_test
        # print(X_batch)  # 测试用
        # print(y_batch)  # 测试用
        # X大写，y小写
        sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        pred_labels = sess.run(logits, feed_dict={X: X_test})
        # print(pred_labels.shape)
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        # pred_labels = sess.run(feed_dict={X: X_test})
        # print(pred_labels)
        acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})

        # 每次打印一下当前信息
        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
