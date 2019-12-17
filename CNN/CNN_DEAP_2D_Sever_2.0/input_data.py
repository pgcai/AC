import scipy.io as scio
import gc
import numpy as np
import random
import pickle
seed = 2
PEOPEL_NUM = 1
DATA_ALL = 40 * PEOPEL_NUM
BATCH_SIZE_TEST = DATA_ALL/4  # 测试数据量
BATCH_SIZE = DATA_ALL - BATCH_SIZE_TEST  # 训练数据量
filepath = '/home/superlee/CC/dataset/CNN_train.pkl'

# 一、1.读取.csv文件


def read_train_data():
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    signal_data = data['data']
    labels = data['labels']
    print("文件已读取！")
    j = 1  # 要取标签的通道
    for i in range(labels.shape[0]):
        if (i + r) % 4 == 0:
            r += 1
        data_list[i] = np.array(data[i + r]).reshape(40, 8064)
        # print(data_list[i])  # 测试
        labels_list[i] = np.array(labels[i+r]).reshape(4)
        labels_list_re = [0]*2
        # print(labels_list[i][j])  # 测试
        if labels_list[i][j] >= 5:
            labels_list_re[0] = 1
            labels_list_re[1] = 0
        else:
            labels_list_re[0] = 0
            labels_list_re[1] = 1
        # print(labels_list_re)  # 测试
        labels_list[i] = np.array(labels_list_re).reshape(2)
    return signal_data, labels  # 包括data（40*40*8064）和labels（40*4)


def data_reshape(data, labels):
    print("终于开始读入训练数据了-_-!")
    j = 1  # 要取标签的通道
    data_list = [0] * int(BATCH_SIZE)
    labels_list = [0] * int(BATCH_SIZE)
    r = 0
    for i in range(int(BATCH_SIZE)):
        if (i + r) % 4 == 0:
            r += 1
        data_list[i] = np.array(data[i + r]).reshape(40, 8064)
        # print(data_list[i])  # 测试
        labels_list[i] = np.array(labels[i+r]).reshape(4)
        labels_list_re = [0]*2
        # print(labels_list[i][j])  # 测试
        if labels_list[i][j] >= 5:
            labels_list_re[0] = 1
            labels_list_re[1] = 0
        else:
            labels_list_re[0] = 0
            labels_list_re[1] = 1
        # print(labels_list_re)  # 测试
        labels_list[i] = np.array(labels_list_re).reshape(2)
        # for j in range(4):
        #     # print(labels_list[i][j])  # 测试
        #     if (labels_list[i][j] >= 5):  # 二化
        #         labels_list[i][j] = 1
        #     else:
        #         labels_list[i][j] = 0
        # print(labels_list[i][j])
        # print(data_list[i])  # 测试用
    print("已读取第{}个训练数据集".format(i))
    print("训练数据读取完成")
    data_list_array = np.array(data_list)
    labels_list_array = np.array(labels_list)
    return data_list_array, labels_list_array  # 返回整理好数据

# 处理测试数据


def data_reshape_test(data, labels):
    j = 1  # 要取标签的通道
    data_list = [0] * int(BATCH_SIZE_TEST)
    labels_list = [0] * int(BATCH_SIZE_TEST)
    for i in range(int(BATCH_SIZE_TEST)):
        data_list[i] = np.array(data[i*4]).reshape(40, 8064)
        labels_list[i] = np.array(labels[i*4]).reshape(4)
        labels_list_re = [0] * 2
        if labels_list[i][j] >= 5:
            labels_list_re[0] = 1
            labels_list_re[1] = 0
        else:
            labels_list_re[0] = 0
            labels_list_re[1] = 1
        print(labels_list_re)  # 测试
        labels_list[i] = np.array(labels_list_re).reshape(2)
    print("测试数据读取完成")
    data_list_array = np.array(data_list)
    labels_list_array = np.array(labels_list)
    return data_list_array, labels_list_array  # 返回整理好数据

