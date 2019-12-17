import math
import numpy as np
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt
class FrequencyDomainFeatures:
    # 求频域特征
    def __init__(self):
        pass

    def FFT(self, signal):
        # 进行傅里叶变换
        result = fft(signal)  # 快速傅里叶变换
        real = result.real  # 获取实数部分
        imag = result.imag  # 获取虚数部分

        result2 = abs(fft(signal))  # 取绝对值
        result2 = result2/len(signal)  # 归一化处理
        return result2[range(int(len(signal)/2))]

