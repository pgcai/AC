# 将信号分段以及按不同类型波分开

import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from scipy import signal


# def readFile(file_path):
#     signal_data = []
#     signal_labels = []
#     for i in range(1, PEOPEL_NUM+1):
#         data = scio.loadmat(file_path + "s" + str(int(i)).zfill(2) + ".mat")  # 读取mat文件数据
#         signal_data += data['data'].tolist()  # data数据 将数据以列表形式作为值，data作为键 的字典格式传递给signal_data
#         signal_labels += data['labels'].tolist()  # labels数据 将标签数据以列表形式作为值，labels作为键 的字典格式传递给signal_labels
#         print("第{}个文件已被读取".format(i))
#     signal_data_re = np.array(signal_data).reshape(DATA_ALL, 40, 8064)
#     print("data原始数据文件成功整形！")
#     signal_labels_re = np.array(signal_labels).reshape(DATA_ALL, 4)
#     print("labels原始数据文件成功整形！")


def readSignal(file_path):
    data = scio.loadmat(file_path + "/" + str(0) + ".mat")
    signaldata = data['A'].tolist()
    signaldata = np.array(signaldata)
    return signaldata


def main():
    filepath = 'F:/情感计算/数据集/EEG_cai'
    signalData = readSignal(filepath)
    # print(signalData)
    print(signalData.shape)
    data = signalData.T
    print(data.shape)
    # 画图
    fig = plt.figure(figsize=[40, 4])  # 设置图大小
    fig = plt.plot(data[500:800, 1:2], label="A")
    data2 = data[500:800, 1:2].T
    print(data2.shape)
    data_b = bandPass(data2, 14, 30, 128)
    fig2 = plt.figure(figsize=[40, 4])  # 设置图大小
    fig2 = plt.plot(data_b.T, label="B")
    # fig3 = plt.figure(figsize=[40, 4])  # 设置图大小
    # fig3 = plt.plot(data[384:640, 2:3], label="C")
    plt.show()


# 带通滤波器
# 信号，频率下限，频率上限， 采样率
def bandPass(signals, fre_low, fre_high, fs):
    b, a = signal.butter(8, [2.0 * fre_low / fs, 2.0 * fre_high / fs], 'bandpass', 'sos')
    filtedData = signal.filtfilt(b, a, signals)
    return filtedData


if __name__ == '__main__':
    main()