import math
import numpy

# 求信号的时域特征
class TimeDomainFeatures:

    def __init__(self):
        self.signal = []  # 传入的信号

    def get_signal(self, signal):
        self.signal = signal

    # 求信号均值，参数为一维数组
    def getMean(self, signal):
        sum = 0
        for i in range(len(signal)):
            sum += signal[i]
        avg = sum / len(signal)
        return avg

    # 求最大值
    def getMax(self, signal):
        return max(signal)

    # 求最小值
    def getMin(self, signal):
        return min(signal)

    # 求方差
    def getVariance(self, signal):
        avg = self.getMean(signal)  # 先求平均值
        # 求方差
        sum_ = 0
        for i in range(len(signal)):
            sum_ += (signal[i] - avg) * (signal[i] - avg)
        variance = sum_ / len(signal)  # 方差
        return variance

    # 求标准差
    def getStandardDeviation(self, signal):
        variance = self.getVariance(signal)  # 方差
        standarddeviation = math.sqrt(variance)  # 标准差
        return standarddeviation

    # 求峰峰值
    def getPeak_to_peak(self, signal):
        max_ = self.getMax(signal)
        min_ = self.getMin(signal)
        peak_to_peak = max_ - min_
        return peak_to_peak

    # 求过均值点个数
    def getNumberofOverMean(self, signal):
        mean = self.getMean(signal)  # 均值
        count = 0
        for i in range(len(signal)):
            if signal[i] > mean:  # 如果该值大于均值，计数器加一
                count += 1

        return count  # 返回一个整型，过均值点个数

    # 求单个半峰的位置(两个半峰为一个峰)
    def getPositionofPeak(self, signal, start, size):
        # 参数分别为 一维数组信号，起始位置, 敏感度（值域）
        # 返回值为，起始值与终点值，最大值与最小值
        # 只要出现拐点就算一个峰，单小于敏感度范围的不算，最后记录 峰结束的位置 以及 该峰的距离大小
        oldPosition = start  # 临时变量记录当前位置
        oldTendency = 0  # 临时变量记录走势，-1为下降， 1位上升
        # 设置走势初始值
        if  start + 1 < len(signal) and signal[start] <= signal[start+1] :
            oldTendency = 1
        else: oldTendency = -1
        for i in range(start, len(signal)): # 一直循环到找到拐点
            # 怎么判断拐点： 临时变量存储当前位置(i)，临时变量记录走势(tendency)，当本次循环走势与之前走势不同时，为拐点
            tendency = 0
            if i + 1 < len(signal) and signal[i] <= signal[i + 1]:
                tendency = 1
            else:
                tendency = -1
            if tendency != oldTendency or i == len(signal)-1:  # 如果与之前的趋势不符，说明出现了拐点,临时变量记录位置
                position = i
                distance = position - oldPosition
                return [[distance, start, position],
                        [max(signal[start: position]), min(signal[start, position])]
                        ]

    # 求周期数
    def getPeriod(self, signal):
        # 传入参数为一维数组，直接用
        peak_to_peak = self.getPeak_to_peak(signal)  # 求峰峰值
        size = peak_to_peak * 0.6  # 定义敏感度，即多大的峰可以被判定


