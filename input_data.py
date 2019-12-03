import numpy as np
import pickle
import matplotlib.pyplot as plt

BATCH_SIZE = 30
channel = 0  # 训练标签通道
testPeople = 32  # 第x个人做测试集


def readFile(file_path):
    with open(file_path, 'rb') as f:
        read_data = pickle.load(f)
    data = read_data['data']
    labels = read_data['labels']

    return data, labels


def train_data(data):
    return data[0:31*40]  # 留一法 先取前31人数据


def train_labels(labels):
    labels_final = []
    for i in range(0, 31*40):
        if labels[i][channel] >= 5:
            labels_final.append([1, 0])
        else:
            labels_final.append([0, 1])
    labels_final = np.array(labels_final)
    return labels_final


def test_data(data):
    return data[(testPeople-1)*40:testPeople*40]  # 留一法 取最后1人数据


def test_labels(labels):
    labels_final = []
    for i in range((testPeople-1)*40, testPeople*40):
        if labels[i][channel] >= 5:
            labels_final.append([1, 0])
        else:
            labels_final.append([0, 1])
    labels_final = np.array(labels_final)
    return labels_final


def main():
    filepath = 'F:/情感计算/Results/PicCut_3.pkl'
    data, labels = readFile(filepath)
    print('--------------data.shape--------------')
    print(data.shape)
    print('--------------labels.shape------------')
    print(labels.shape)
    # print(labels)
    train_d = train_data(data)
    train_l = train_labels(labels)

    print('--------------train_d.shape--------------')
    print(train_d.shape)
    print('--------------train_l.shape------------')
    print(train_l.shape)

    # ----------------------------------------------
    # 单个图片输出测试
    # x = train_d[1, :, :, 0:3]
    # print(x.shape)
    # plt.imshow(x)
    # plt.show()
    # ----------------------------------------------

    # print(train_l)
    test_d = test_data(data)
    test_l = test_labels(labels)
    print('--------------test_d.shape--------------')
    print(test_d.shape)
    print('--------------test_l.shape------------')
    print(test_l.shape)
    print(test_l)


if __name__ == '__main__':
    main()



