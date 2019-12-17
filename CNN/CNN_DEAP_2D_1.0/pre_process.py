import numpy as np
np.random.seed(2)


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
    more_norm_dataset_1D = np.zeros([dataset_1D.shape[0], 8064, 32])
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
    dataset_2D = np.zeros([dataset_1D.shape[0], 8064, 9, 9])
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


def main():
    # filepath1 = filepath()
    # signal_data, signal_labels = readFile(filepath1)
    # # (1280, 40, 8064)
    # # (1280, 4)
    # re_data = pre_data_reshape(signal_data)
    # # print(re_data)
    # # print(re_data.shape)
    # z_score_data = more_norm_dataset(re_data)
    # print(z_score_data.shape)
    # data_1Dto2D = more_dataset_1Dto2D(z_score_data)
    # print(data_1Dto2D.shape)
    # # print(data_1Dto2D)
    print("该程序为预处理程序")


if __name__ == '__main__':
    main()

