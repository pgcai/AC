# coding:utf-8
import tensorflow as tf
import forward
import os
import numpy as np
from pre_process import pre_data_reshape, more_norm_dataset, more_dataset_1Dto2D
from input_data import readFile, data_reshape
import input_data
import pickle
# from input_data import readFile, data_reshape
# import cPickle
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 屏蔽gpu warning用
# 默认为0：输出所有log信息
# 设置为1：进一步屏蔽INFO信息
# 设置为2：进一步屏蔽WARNING信息
# 设置为3：进一步屏蔽ERROR信息

# 定义训练过程中的超参数
BATCH_SIZE = 10  # 一个batch的数量
PEOPEL_NUM = input_data.PEOPEL_NUM
BATCH_SIZE_ALL = PEOPEL_NUM*40//4*3
LEARNING_RATE_BASE = 0.0001  # 初始学习率
LEARNING_RATE_DECAY = 0.99  # 学习率的衰减率
REGULARIZER = 0.00001  # 正则化项的权重
STEPS = 100000  # 最大迭代次数
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均的衰减率
MODEL_SAVE_PATH = "./model/"  # 保存模型的路径
MODEL_NAME = "deap_2D_model"  # 模型命名
filepath = 'F:/情感计算/数据集/DEAP/'
# 训练过程


def backward(signal_re, labels_re):
    # x,y_是定义的占位符,需要指定参数的类型,维度(要和网络的输入与输出维度一致),类似于函数的形参,运行时必须输入的参数
    x = tf.placeholder(tf.float32,
                       [BATCH_SIZE,
                        forward.IMAGE_SIZE_X,
                        forward.IMAGE_SIZE_Y,
                        forward.NUM_CHANNELS])
    y_ = tf.placeholder(tf.float32, [None, forward.OUTPUT_NODE])

    # 调用前向传播网络得到维度为8的tensor
    y = forward.forward(x, True, REGULARIZER)
    # 声明一个全局计数器,并输出化为0 并说明是不参与训练过程的
    global_step = tf.Variable(0, trainable=False)

    # 先是对网络最后一层的输出y做softmax,通常是求取输出属于某一类的概率,其实就是num_classes的大小的向量
    # 再将此向量和实际标签值做交叉熵,需要说明的是该函数的返回是一个向量
    # ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))  # 不会用 暂放
    # ce = tf.square(y - y_)  # 有问题
    # 再对得到的向量求均值就得到loss
    cem = tf.reduce_mean(tf.square(y - y_))
    loss = cem + tf.add_n(tf.get_collection('losses'))  # 添加正则化中的losses

    # 实现指数级的减小学习率,可以让模型在训练的前期快速接近较优解，又可以保证模型在训练后期不会有太大波动
    # 计算公式:decayed_learning_rate = learning_rate*decay_rate^(global_step/decay_steps)
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        BATCH_SIZE_ALL / BATCH_SIZE,  # 每几轮改变一次
        LEARNING_RATE_DECAY,
        staircase=True)  # True(global_step/decay_steps)取整,阶梯 False 平滑

    # 传入学习率,构造一个实现梯度下降算法的优化器,再通过使用minimize更新存储要训练的变量列表来减少loss
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 实现滑动平均模型,参数MOVING_AVERAGE_DECAY用于控制模型更新速度.训练过程中会对每一个变量维护一个影子变量,这个影子变量的初始值
    # 就是相应变量的初始值,每次变量更新时,影子变量就会随之更新
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]):  # 将train_step和eam_op两个训练操作绑定到train_op上
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()  # 实例化一个保存和恢复变量的saver

    # 创建一个会话,并通过python中的上下文管理器来管理这个会话
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()  # 初始化计算图中的变量
        sess.run(init_op)

        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)  # 通过checkpoint文件定位到最新保存的模型
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)  # 加载最新的模型

        for i in range(STEPS):
            start = (i * BATCH_SIZE) % BATCH_SIZE_ALL
            end = start + BATCH_SIZE
            xs = signal_re
            ys = labels_re[start:end]  # 读取一个batch的数据
            reshaped_xs = np.reshape(  # 将输入数据xs转换成与网络输入相同形状的矩阵
                xs[start:end],  # 读取一个batch的数据
                (BATCH_SIZE,
                 forward.IMAGE_SIZE_X,
                 forward.IMAGE_SIZE_Y,
                 forward.NUM_CHANNELS))
            # 喂入训练图像&标签, 开始训练
            # print(reshaped_xs)  # debug
            # print(ys)  # debug
            # print("-------------我是分割线-------------")  # debug用
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys})
            # print("-------------我是分割线-------------")  # debug用
            if i % 1 == 0:  # 每迭代10次打印loss信息,20000次保存最新的模型
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
            if i % 10 == 0 and i != 0:
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
                print("After %d training step(s), Model has been saved." % step)


def main():
    # 读取原始文件数据，标签
    print("-------读取原始文件数据，标签--------")
    signal_data, signal_labels = readFile(filepath)
    # 将数据和标签的形状进行调整
    print("-----将数据和标签的形状进行调整------")
    signal_re, labels_re = data_reshape(signal_data, signal_labels)
    re_data = pre_data_reshape(signal_re)
    # 数据进行Z-score
    print("----------数据进行Z-score----------")
    z_score_data = more_norm_dataset(re_data)
    # 数据进行1D->2D的转化
    print("--------数据进行1D->2D的转化--------")
    data_1Dto2D = more_dataset_1Dto2D(z_score_data)
    print(data_1Dto2D.shape)
    dict_data = {"data": data_1Dto2D, "labels":labels_re}
    with open('CNN_train.pkl', 'wb') as f:
        pickle.dump(dict_data, f, pickle.HIGHEST_PROTOCOL)
    # 读取.pkl文件
    # with open('CNN_train.pkl', 'rb') as f:
    #     data =  pickle.load(f)
    # data_1Dto2D = data['data']
    # labels_re = data['labels']
    # 开始CNN的训练
    print("-----------开始CNN的训练-----------")
    backward(data_1Dto2D, labels_re)


if __name__ == '__main__':
    main()


