# coding:utf-8
import tensorflow as tf
# from tensorflow.contrib.layers import xavier_initializer  # 一种随机值
# 设定神经网络的超参数
# 定义神经网络可以接受的矩阵的尺寸和通道数
IMAGE_SIZE_X = 63
IMAGE_SIZE_Y = 128
NUM_CHANNELS = 40
# 定义第一层的卷积核大小和个数
CONV1_SIZE = 5
CONV1_KERNEL_NUM = 64
# 定义第二层的卷积核大小和个数
CONV2_SIZE = 5
CONV2_KERNEL_NUM = 128
# 定义第三层的卷积核大小和个数
CONV3_SIZE = 5
CONV3_KERNEL_NUM = 128
# 定义第三层全连接层的神经元个数
FC_SIZE = 2048
# 定义第四层全连接层的神经元个数
OUTPUT_NODE = 8

# 定义初始化网络权重函数


def get_weight(shape, regularizer):
    # shape生成张量维度  regularizer正则化项的权重
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    # tf.truncated_normal生成去掉过大偏离点的正态分布随机数的张量；stddev 是指定标准差
    if regularizer != None: tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    # 为权重加入L2正则化，通过限制权重大小使模型不会随意拟合训练数据中的随机噪音
    return w

# 定义初始化偏置项函数


def get_bias(shape): 
    b = tf.Variable(tf.zeros(shape))
    # 统一将bias初始化为0
    return b

# 定义卷积计算函数


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
    # x：一个输入batch(批量) w：卷积层的权重 strides=[1,行步长,列步长,1] padding='SAME'全零填充 'VALID'不填充


def max_pool_2x2(x):  
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # x：一个输入batch(批量) ksize表示池化过滤器的边长为2 strides表示过滤器移动步长 padding='SAME'全零填充 'VALID'不填充

# 定义前向传播过程


def forward(x, train, regularizer):
    # x:一个输入batch train:用于区分训练过程True，测试过程False regularizer:正则化权重
    # 实现第一层卷积层的前向传播过程
    # 初始化卷积核
    conv1_w = get_weight([CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_KERNEL_NUM], regularizer)
    # 5 5 40 64
    # 初始化偏置项
    conv1_b = get_bias([CONV1_KERNEL_NUM])
    # 64
    # 实现卷积运算
    conv1 = conv2d(x, conv1_w)
    # 61 * 126 * 64
    # 对卷积后的输出添加偏置，并通过relu非线性激活函数
    relu1 = tf.nn.sigmoid(tf.nn.bias_add(conv1, conv1_b))
    # 将激活后的输出进行最大池化
    pool1 = max_pool_2x2(relu1)
    # print("666666666666666666666666666666666666666666666666")  # debug用

    # 实现第二层卷积层的前向传播过程，并初始化卷积层的相对应变量
    # 该层每个卷积核的通道数要与上一层卷积核的个数一致
    conv2_w = get_weight([CONV2_SIZE, CONV2_SIZE, CONV1_KERNEL_NUM, CONV2_KERNEL_NUM], regularizer)
    conv2_b = get_bias([CONV2_KERNEL_NUM])
    # 该层的输入就是上一层的输出pool1
    conv2 = conv2d(pool1, conv2_w)
    relu2 = tf.nn.sigmoid(tf.nn.bias_add(conv2, conv2_b))
    pool2 = max_pool_2x2(relu2)

    # 实现第三层卷积层的前向传播过程，并初始化卷积层的相对应变量
    # 该层每个卷积核的通道数要与上一层卷积核的个数一致
    conv3_w = get_weight([CONV3_SIZE, CONV3_SIZE, CONV2_KERNEL_NUM, CONV3_KERNEL_NUM], regularizer)
    conv3_b = get_bias([CONV3_KERNEL_NUM])
    # 该层的输入就是上一层的输出pool1
    conv3 = conv2d(pool2, conv3_w)
    relu3 = tf.nn.sigmoid(tf.nn.bias_add(conv3, conv3_b))
    pool3 = max_pool_2x2(relu3)

    # 将上一池化层的输出pool2(矩阵)转化为下一层全连接的输入格式(向量)
    pool_shape = pool3.get_shape().as_list()
    # 得到pool2输出矩阵维度,并存入list中,注意pool_shape[0]是一个batch值
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]  # 依次取出长宽深，并求三者乘积即为矩阵拉长后长度
    reshaped = tf.reshape(pool3, [pool_shape[0], nodes])  # 将pool2转换为一个batch的向量再传入后续的全连接

    # 实现第三层全连接层的前向传播过程
    fc1_w = get_weight([nodes, FC_SIZE], regularizer)  # 初始化全连接层的权重,并加入正则化
    fc1_b = get_bias([FC_SIZE])  # 初始化全连接层的偏置项
    # 将转换后的reshaped向量与权重fc1_w做矩阵乘法运算,然后再加上偏置,最后再使用relu进行激活
    fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_w) + fc1_b)
    # 如果是训练阶段，则对该层输出使用dropout,也就是随机的将该层的输出中的一半神经元置为无效,
    # 是为了避免过拟合而设置的,一般只在全连接层中使用
    if train:
        fc1 = tf.nn.dropout(fc1, 0.5)

    # 实现第四层全连接层的前向传播过程，并初始化全连接层对应的变量
    fc2_w = get_weight([FC_SIZE, OUTPUT_NODE], regularizer)
    fc2_b = get_bias([OUTPUT_NODE])
    y = tf.matmul(fc1, fc2_w) + fc2_b
    return y 
