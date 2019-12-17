import scipy.signal as signal
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import matplotlib
from scipy import signal
#过滤
class Filter:
    #  滤波
    def __init__(self):
        self.signal = []  # 传入的信号

    def get_signal(self, signal):
        self.signal = signal

    def medianFilter(self, signal, n):
        # 中位值平均滤波法
        # signal是需要进行滤波的全部信号
        y_next = []
        tempY = []
        k = 0
        for i in range(len(signal)):
            tempY.append(signal[i])
            if i % n == 0:  # 每n个数进行一次滤波处理

                ###############################################
                Length = len(tempY)

                # 数据从小到大排序
                for i in range(Length - 1):
                    for j in range(Length - 1 - i):
                        if tempY[j] > tempY[j + 1]:
                            temp = tempY[j]
                            tempY[j] = tempY[j + 1]
                            tempY[j + 1] = temp

                # 去除最大最小值后求平均值
                filter_sum = 0
                for i in range(1, Length - 1):
                    filter_sum += tempY[i]
                t = filter_sum // (Length - 2)
                ##############################################3
                y_next.append(t)
                k += 1
                tempY = []
        signal = y_next
        return signal  # 返回新的、已滤波的信号


# 去基线漂移
def takeBaselineExcursion(signals, fre1, fre2, sampling_fre, type):
    # 参数为：信号【一维数组】，频率【以多高的频率为界限】，采样率， 方式【是高通、低通还是带通，如果为带通，传入两个fre，否则fre2为0】
    #  wn = 2 * 界值频率 / 采样率
    wn1 = 2.0 * fre1 / sampling_fre
    wn2 = 0
    b = 0
    a = 0
    if type == "bandpass":
        wn2 = 2.0 * fre2 / sampling_fre

    if type == "highpass":
        b, a = signal.butter(8, wn1, type)
    elif type == "lowpass":
        b, a = signal.butter(8, wn1, type)
    elif type == "bandpass":
        b, a = signal.butter(8, [wn1, wn2], type)

    result = signal.filtfilt(b, a, signals)
    return result


# 带通滤波器
# 信号，频率下限，频率上限， 采样率
def bandPass(signals, fre_low, fre_high, fs):
    b, a = signal.butter(8, [2.0 * fre_low / fs, 2.0 * fre_high / fs], 'bandpass')
    filtedData = signal.filtfilt(b, a, signals)
    return filtedData

# 低通滤波器
def lowPass(signals, fre, fs):
    b, a = signal.butter(8, 2.0 * fre / fs, 'lowpass')
    filtedData = signal.filtfilt(b, a, signals)
    return filtedData

# 高通滤波器
def highPass(signals, fre, fs):
    b, a = signal.butter(8, 2.0 * fre / fs, 'highpass')
    filtedData = signal.filtfilt(b, a, signals)
    return filtedData


'''
算术平均滤波法
'''


def ArithmeticAverage(inputs, per):
    if np.shape(inputs)[0] % per != 0:
        lengh = np.shape(inputs)[0] / per
        for x in range(int(np.shape(inputs)[0]), int(lengh + 1) * per):
            inputs = np.append(inputs, inputs[np.shape(inputs)[0] - 1])
    inputs = inputs.reshape((-1, per))
    mean = []
    for tmp in inputs:
        mean.append(tmp.mean())
    return mean


'''
递推平均滤波法
'''


def SlidingAverage(inputs, per):
    if np.shape(inputs)[0] % per != 0:
        lengh = np.shape(inputs)[0] / per
        for x in range(int(np.shape(inputs)[0]), int(lengh + 1) * per):
            inputs = np.append(inputs, inputs[np.shape(inputs)[0] - 1])
    inputs = inputs.reshape((-1, per))
    tmpmean = inputs[0].mean()
    mean = []
    for tmp in inputs:
        mean.append((tmpmean + tmp.mean()) / 2)
        tmpmean = tmp.mean()
    return mean


'''
中位值平均滤波法
'''


def MedianAverage(inputs, per):
    if np.shape(inputs)[0] % per != 0:
        lengh = np.shape(inputs)[0] / per
        for x in range(int(np.shape(inputs)[0]), int(lengh + 1) * per):
            inputs = np.append(inputs, inputs[np.shape(inputs)[0] - 1])
    inputs = inputs.reshape((-1, per))
    mean = []
    for tmp in inputs:
        tmp = np.delete(tmp, np.where(tmp == tmp.max())[0], axis=0)
        tmp = np.delete(tmp, np.where(tmp == tmp.min())[0], axis=0)
        mean.append(tmp.mean())
    return mean


'''
限幅平均滤波法
Amplitude:	限制最大振幅
'''


def AmplitudeLimitingAverage(inputs, per, Amplitude):
    if np.shape(inputs)[0] % per != 0:
        lengh = np.shape(inputs)[0] / per
        for x in range(int(np.shape(inputs)[0]), int(lengh + 1) * per):
            inputs = np.append(inputs, inputs[np.shape(inputs)[0] - 1])
    inputs = inputs.reshape((-1, per))
    mean = []
    tmpmean = inputs[0].mean()
    tmpnum = inputs[0][0]  # 上一次限幅后结果
    for tmp in inputs:
        for index, newtmp in enumerate(tmp):
            if np.abs(tmpnum - newtmp) > Amplitude:
                tmp[index] = tmpnum
            tmpnum = newtmp
        mean.append((tmpmean + tmp.mean()) / 2)
        tmpmean = tmp.mean()
    return mean


'''
一阶滞后滤波法
a:			滞后程度决定因子，0~1
'''


def FirstOrderLag(inputs, a):
    tmpnum = inputs[0]  # 上一次滤波结果
    for index, tmp in enumerate(inputs):
        inputs[index] = (1 - a) * tmp + a * tmpnum
        tmpnum = tmp
    return inputs


'''
加权递推平均滤波法
'''


def WeightBackstepAverage(inputs, per):
    weight = np.array(range(1, np.shape(inputs)[0] + 1))  # 权值列表
    weight = weight / weight.sum()

    for index, tmp in enumerate(inputs):
        inputs[index] = inputs[index] * weight[index]
    return inputs


'''
消抖滤波法
N:			消抖上限
'''


def ShakeOff(inputs, N):
    usenum = inputs[0]  # 有效值
    i = 0  # 标记计数器
    for index, tmp in enumerate(inputs):
        if tmp != usenum:
            i = i + 1
            if i >= N:
                i = 0
                inputs[index] = usenum
    return inputs


'''
限幅消抖滤波法
Amplitude:	限制最大振幅
N:			消抖上限
'''


def AmplitudeLimitingShakeOff(inputs, Amplitude, N):
    # print(inputs)
    tmpnum = inputs[0]
    for index, newtmp in enumerate(inputs):
        if np.abs(tmpnum - newtmp) > Amplitude:
            inputs[index] = tmpnum
        tmpnum = newtmp
    # print(inputs)
    usenum = inputs[0]
    i = 0
    for index2, tmp2 in enumerate(inputs):
        if tmp2 != usenum:
            i = i + 1
            if i >= N:
                i = 0
                inputs[index2] = usenum
    # print(inputs)
    return inputs
