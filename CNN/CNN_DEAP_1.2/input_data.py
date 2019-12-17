import scipy.io as scio
import gc
import numpy as np
import random

BATCH_SIZE = 30  # 训练数据量
BATCH_SIZE_TEST = 10  # 测试数据量

# 一、1.读取.csv文件


def readFile(file_path):
    data = scio.loadmat(file_path)  # 读取mat文件数据
    signal_data = data['data'].tolist()  # data数据 将数据以列表形式作为值，data作为键 的字典格式传递给signal_data
    signal_labels = data['labels'].tolist()  # labels数据 将标签数据以列表形式作为值，labels作为键 的字典格式传递给signal_labels
    # ------------
    # 手动释放内存
    del data
    gc.collect()
    return signal_data, signal_labels  # 包括data（40*40*8064）和labels（40*4)

# 处理输入训练数据


def data_reshape(data, labels):
    data_list = [0] * BATCH_SIZE
    labels_list = [0] * BATCH_SIZE
    r = 0
    for i in range(BATCH_SIZE):
        if (i + r) % 4 == 0:
            r += 1
        data_list[i] = np.array(data[i + r]).reshape(40, 63, 128)
        labels_list[i] = np.array(labels[i + r]).reshape(4)
        print(labels_list[i])
        # if (labels_list[i][0] >= 5):  # 二化
        #     labels_list[i] = [1]
        # else:
        #     labels_list[i] = [0]
        # print(labels_list[i])
        # print(data_list[i])  # 测试用
    return data_list, labels_list  # 返回整理好数据

# 处理测试数据


def data_reshape_test(data, labels):
    data_list = [0] * BATCH_SIZE_TEST
    labels_list = [0] * BATCH_SIZE_TEST
    for i in range(BATCH_SIZE_TEST):
        data_list[i] = np.array(data[i*4]).reshape(40, 63, 128)
        labels_list[i] = np.array([labels[i*4]]).reshape(4)
        # if (labels_list[i][0] >= 5):  # 二化
        #     labels_list[i] = [1]
        # else:
        #     labels_list[i] = [0]
        print(labels_list[i])  # 测试用
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



