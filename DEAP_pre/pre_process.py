import numpy as np
import scipy.io as scio
import gc
import pickle
np.random.seed(2)

seed = 2
PEOPEL_NUM = 32  # 人数
DATA_ALL = 40 * PEOPEL_NUM
# BATCH_SIZE_TEST = DATA_ALL/4  # 测试数据量
# BATCH_SIZE = DATA_ALL - BATCH_SIZE_TEST  # 训练数据量
filepath = 'F:/情感计算/数据集/DEAP/'
# filepath = '/mnt/external/superlee/dataset/DEAP_dataset/data_preprocessed_matlab/'


def readFile(file_path):
    signal_data = []
    signal_labels = []
    for i in range(1, PEOPEL_NUM+1):
        data = scio.loadmat(file_path + "s" + str(int(i)).zfill(2) + ".mat")  # 读取mat文件数据
        signal_data += data['data'].tolist()  # data数据 将数据以列表形式作为值，data作为键 的字典格式传递给signal_data
        signal_labels += data['labels'].tolist()  # labels数据 将标签数据以列表形式作为值，labels作为键 的字典格式传递给signal_labels
        print("第{}个文件已被读取".format(i))
    signal_data_re = np.array(signal_data).reshape(DATA_ALL, 40, 8064)
    print("data原始数据文件成功整形！")
    signal_labels_re = np.array(signal_labels).reshape(DATA_ALL, 4)
    print("labels原始数据文件成功整形！")

    # 手动释放内存
    del data
    gc.collect()
    return signal_data_re, signal_labels_re  # 包括data（40*40*8064）和labels（40*4)


def feature_normalize(data):  # Z-score
    mean = data[data.nonzero()].mean()
    sigma = data[data. nonzero()].std()
    data_normalized = data
    data_normalized[data_normalized.nonzero()] = (data_normalized[data_normalized.nonzero()] - mean)/sigma
    # return shape: 9*9
    return data_normalized


def norm_dataset(dataset_1D):  # onePerson-Z-score
    norm_dataset_1D = np.zeros([dataset_1D.shape[0], 32])
    for i in range(dataset_1D.shape[0]):
        norm_dataset_1D[i] = feature_normalize(dataset_1D[i])
    # return shape: m*32
    return norm_dataset_1D


def more_norm_dataset(dataset_1D):  # morePerson-Z-score
    print("Z-score Began*** ***")
    more_norm_dataset_1D = np.zeros([dataset_1D.shape[0], dataset_1D.shape[1], 32])
    for i in range(dataset_1D.shape[0]):
        more_norm_dataset_1D[i] = norm_dataset(dataset_1D[i])
    # return shape: m*32
    return more_norm_dataset_1D


def one_data_1Dto2D(data, Y=9, X=9):  # 1Dto2D
    data_2D = np.zeros([Y, X])
    data_2D[0] = (0, 0, 0, data[0], 0, data[16], 0, 0, 0)
    data_2D[1] = (0, 0, 0, data[1], 0, data[17], 0, 0, 0)
    data_2D[2] = (data[3], 0, data[2], 0, data[18], 0, data[19], 0, data[20])
    data_2D[3] = (0, data[4], 0, data[5], 0, data[22], 0, data[21], 0)
    data_2D[4] = (data[7], 0, data[6], 0, data[23], 0, data[24], 0, data[25])
    data_2D[5] = (0, data[8], 0, data[9], 0, data[27], 0, data[26], 0)
    data_2D[6] = (data[11], 0, data[10], 0, data[15], 0, data[28], 0, data[29])
    data_2D[7] = (0, 0, 0, data[12], 0, data[30], 0, 0, 0)
    data_2D[8] = (0, 0, 0, data[13], data[14], data[31], 0, 0, 0)
    # return shape:9*9
    return data_2D


def dataset_1Dto2D(dataset_1D):  # onePerson_batch-1Dto2D
    dataset_2D = np.zeros([dataset_1D.shape[0], 9, 9])
    for i in range(dataset_1D.shape[0]):
        dataset_2D[i] = one_data_1Dto2D(dataset_1D[i])
    # return shape: m*9*9
    return dataset_2D


def more_dataset_1Dto2D(dataset_1D):  # morePerson_batch-1Dto2D
    print("1D to 2D Began*** ***")
    dataset_2D = np.zeros([dataset_1D.shape[0], dataset_1D.shape[1], 9, 9])
    for i in range(dataset_1D.shape[0]):
        dataset_2D[i] = dataset_1Dto2D(dataset_1D[i])
    # return shape: m*9*9
    return dataset_2D


def pre_data_reshape(signal_data):  # (*,8064) to (*/40, 8064, 32)
    re_data = np.empty([signal_data.shape[0], 8064, 32])
    for i in range(signal_data.shape[0]):
        for j in range(8064):
            for k in range(32):
                re_data[i][j][k] = signal_data[i][k][j]
    re_data_np = np.array(re_data).reshape(signal_data.shape[0], 8064, 32)
    return re_data_np


def pre_baseline(signal_data):  # 基线预处理
    all_base = []
    signal_data = signal_data[:, 0:40]  # 将32个人的数据读前32条
    for i in range(signal_data.shape[0]):  # for循环中 一次处理一个人的数据
        one_signal_data = np.hsplit(signal_data[i], 63)  # 将一个人的信号分割成63段
        base_mean = (one_signal_data[0]+one_signal_data[1]+one_signal_data[2])/3  # 求前三段的平均值
        one_raw_base = one_signal_data[3] - base_mean
        one_raw_base = np.array(one_raw_base)
        for j in range(59):
            raw_base = one_signal_data[j + 4] - base_mean
            # raw_base = one_signal_data[j + 4]
            one_raw_base = np.hstack((one_raw_base, raw_base))
            # raw_base_re = np.hsplit(raw_base, 128)
            # for k in range(128):
            #     one_line = raw_base_re[k].reshape(32)
            #     one_line = np.array(one_line)
            #     one_raw_base.append(one_line)
        # one_raw_base = np.array(one_raw_base)
        # print("6666666")
        # print(one_raw_base.shape)
        all_base.append(one_raw_base)
    all_base = np.array(all_base)
    return all_base


def pre_labels(signal_labels):  # 标签预处理
    for i in range(signal_labels.shape[0]):
        # print(signal_labels[i])  # 测试用
        for j in range(4):
            if (signal_labels[i][j] >= 5):  # 二化
                signal_labels[i][j] = 1
            else:
                signal_labels[i][j] = 0
        # print(signal_labels[i])  # 测试用
    labels_array = np.array(signal_labels)
    return labels_array  # 返回整理好数据


def main():
    print("-----------预处理程序开始-----------")
    print("---为方便训练 文件将转储为.pkl文件---")
    print("-------读取原始文件数据，标签--------")
    signal_data, signal_labels = readFile(filepath)

    # one_data = signal_data[0][0:32]
    # print(one_data.shape)
    #
    # savePath = 'F:/情感计算/数据集/EEG_cai/one.mat'
    # scio.savemat(savePath, {'A': one_data})
    #
    #
    print("---------原始数据data.shape---------")
    print(signal_data.shape)
    # 将数据和标签的形状进行调整
    print("---------将信号进行基线预处理--------")
    pre_data = pre_baseline(signal_data)
    print("--------基线处理后data.shape--------")
    # print(pre_data.shape)
    # print("----------批量输出单条数据----------")
    # -------------------------------------------------------
    # # 减基线已注销 按样本输出保存所有信号数据
    # for i in range(pre_data.shape[0]):
    #     savePath = 'F:/情感计算/数据集/EEG_cai/{}.mat'.format(i)
    #     scio.savemat(savePath, {'A': pre_data[i][0:32]})
    #     print(savePath)
    # # 数据进行Z-score
    # print("----------数据进行Z-score----------")
    # z_score_data = more_norm_dataset(pre_data)
    # # 数据进行1D->2D的转化
    # print("--------数据进行1D->2D的转化--------")
    # data_1Dto2D = more_dataset_1Dto2D(z_score_data)
    print("--------正在进行标签的二值化--------")
    # labels_re = pre_labels(signal_labels)
    # print("------------data.shape------------")
    # print(data_1Dto2D.shape)
    # labels_data_out = []
    # ------------------------------------------------------------
    # 输出保存labels
    # print(signal_labels)
    print("labels shape:".format(signal_labels.shape))
    dict_data = {"labels": signal_labels}
    with open('F:/情感计算/Results/labels.pkl', 'wb') as f:
        pickle.dump(dict_data, f, pickle.HIGHEST_PROTOCOL)
    #  ---------------------------------------------------------
    #  输出保存data
    # savePath = 'F:/情感计算/数据集/EEGlab/data.mat'
    # scio.savemat(savePath, {'A': pre_data[:, 0:32, :]})
    #  ---------------------------------------------------------
    # print("------------label.shape------------")
    # print(labels_re.shape)
    # print("---------开始数据的.pkl存储---------")
    # dict_data = {"data": pre_data, "labels": labels_re}
    # with open('/mnt/external/cc/deap_pre.pkl', 'wb') as f:
    #     pickle.dump(dict_data, f, pickle.HIGHEST_PROTOCOL)
    print("--------------存储完成--------------")


if __name__ == '__main__':
    main()

