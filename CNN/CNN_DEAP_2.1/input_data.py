import scipy.io as scio
import gc
import numpy as np
import random

PEOPEL_NUM = 1
DATA_ALL = 40 * PEOPEL_NUM
BATCH_SIZE_TEST = DATA_ALL/4  # 测试数据量
BATCH_SIZE = DATA_ALL - BATCH_SIZE_TEST  # 训练数据量

# 一、1.读取.csv文件


def readFile(file_path):
    signal_data = []
    signal_labels = []
    for i in range(1, PEOPEL_NUM+1):
        data = scio.loadmat(file_path + "s" + str(int(i)).zfill(2) + ".mat")  # 读取mat文件数据
        signal_data += data['data'].tolist()  # data数据 将数据以列表形式作为值，data作为键 的字典格式传递给signal_data
        signal_labels += data['labels'].tolist()  # labels数据 将标签数据以列表形式作为值，labels作为键 的字典格式传递给signal_labels
        print("第{}个文件已被读取".format(i))

    signal_data_re = np.array(signal_data).reshape(DATA_ALL, 40, 63, 128)
    print("data原始数据文件成功整形！")
    signal_labels_re = np.array(signal_labels).reshape(DATA_ALL, 4)
    print("labels原始数据文件成功整形！")

    # ------------
    # 手动释放内存
    del data
    gc.collect()
    return signal_data_re, signal_labels_re  # 包括data（40*40*8064）和labels（40*4)

# 处理输入训练数据


def data_reshape(data, labels):
    print("终于开始读入训练数据了-_-!")
    data_list = [0] * int(BATCH_SIZE)
    labels_list = [0] * int(BATCH_SIZE)
    r = 0
    for i in range(int(BATCH_SIZE)):
        print("已读取第{}个训练数据集".format(i))
        if (i + r) % 4 == 0:
            r += 1
        data_list[i] = np.array(data[i + r]).reshape(40, 63, 128)
        labels_list[i] = np.array(labels[i+r]).reshape(4)
        labels_list_re = [0]*8
        for j in range(4):
            print(labels_list[i][j])  # 测试
            if labels_list[i][j] >= 5:
                labels_list_re[2*j] = 1
                labels_list_re[2 * j+1] = 0
            else:
                labels_list_re[2 * j] = 0
                labels_list_re[2 * j + 1] = 1
        print(labels_list_re)  # 测试
        labels_list[i] = np.array(labels_list_re).reshape(8)
        # for j in range(4):
        #     # print(labels_list[i][j])  # 测试
        #     if (labels_list[i][j] >= 5):  # 二化
        #         labels_list[i][j] = 1
        #     else:
        #         labels_list[i][j] = 0
            # print(labels_list[i][j])
        # print(data_list[i])  # 测试用
    print("训练数据读取完成")
    return data_list, labels_list  # 返回整理好数据

# 处理测试数据


def data_reshape_test(data, labels):
    data_list = [0] * int(BATCH_SIZE_TEST)
    labels_list = [0] * int(BATCH_SIZE_TEST)
    for i in range(int(BATCH_SIZE_TEST)):
        data_list[i] = np.array(data[i*4]).reshape(40, 63, 128)
        labels_list[i] = np.array(labels[i*4]).reshape(4)
        labels_list_re = [0] * 8
        for j in range(4):
            print(labels_list[i][j])  # 测试
            if labels_list[i][j] >= 5:
                labels_list_re[2 * j] = 1
                labels_list_re[2 * j + 1] = 0
            else:
                labels_list_re[2 * j] = 0
                labels_list_re[2 * j + 1] = 1
        print(labels_list_re)  # 测试
        labels_list[i] = np.array(labels_list_re).reshape(8)
    print("测试数据读取完成")
    return data_list, labels_list  # 返回整理好数据

# ---------------------------------------------老版代码----------------------------------------
# def signal_reshape(data):
#     data_list = [0] * BATCH_SIZE
#     for i in range(BATCH_SIZE):  # 共40组
#         # np.array = [y for x in data[i] for y in x]  # 无用
#         data_list[i] =np.array(data[i]).reshape(40, 63, 128)
#         # print(data_list[i])  # 测试用
#     return data_list  # 返回整理好数据


# def labels_reshape(labels):
#     # L 版
#     labels_list = [0] * BATCH_SIZE
#     for i in range(BATCH_SIZE):  # 共40组
#         # labels_list[i] = [labels[i][3]]   # 是否取整
#         if(labels[i][3] > 5):
#             labels_list[i] = [1]
#         else:
#             labels_list[i] = [0]
#         # print(labels_list[i])  # 测试用
#     # print(labels_list)  # debug
#     return labels_list

# 测试组数据整理


# def signal_reshape_test(data):
#     data_list = [0] * 10
#     for i in range(10):  # 共40组
#         # np.array = [y for x in data[i] for y in x]  # 无用
#         data_list[i] =np.array(data[i + random.randint(0, 4)]).reshape(40, 63, 128)
#         # print(data_list[i])  # 测试用
#     return data_list  # 返回整理好数据
#
# # 处理标签


# def labels_reshape_test(labels):
#     # L 版
#     labels_list = [0] * 10
#     for i in range(10):  # 共40组
#         labels_list[i] = [labels[i + random.randint(0, 4)][3]]   # 是否取整
#         # if (labels[i][3] > 5):
#         #     labels_list[i] = [1]
#         # else:
#         #     labels_list[i] = [0]
#         # print(labels_list[i])  # 测试用
#     # print(labels_list)  # debug
#     return labels_list
# ---------------------------------------------老版代码----------------------------------------



