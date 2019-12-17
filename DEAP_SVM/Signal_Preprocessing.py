from Filter import takeBaselineExcursion        # 滤波

# 各个信号预处理(不同信号处理方式不同) 函数
# 一、3.1 EEG去干扰    EEG脑电图  #未处理 待解决
def EEGFilter(signal):
    # signal = takeBaselineExcursion(signal, 3, 0, 128, "lowpass")  # 去高频毛刺，低通 #成功
    # signal = takeBaselineExcursion(signal, 5, 20, 128, "bandpass")  # 去基线漂移后 #暂失败
    return signal


# 一、3.3 EOGh去干扰    EOG眼电图
def EOGhFilter(signal):
    # signal = takeBaselineExcursion(signal, 1, 0, 128, "lowpass")  # 去高频毛刺，低通
    # signal = takeBaselineExcursion(signal, 1, 20, 128, "bandpass")  # 去基线漂移后
    return signal


# 一、3.3 EOGv去干扰    EOG眼电图
def EOGvFilter(signal):
    # signal = takeBaselineExcursion(signal, 1, 0, 128, "lowpass")  # 去高频毛刺，低通
    # signal = takeBaselineExcursion(signal, 1, 20, 128, "bandpass")  # 去基线漂移后
    return signal


# 一、3.4 EMGz去干扰      EMG肌电图
def EMGzFilter(signal):
    # signal = takeBaselineExcursion(signal, 0.7, 0, 128, "lowpass")  # 去高频毛刺，低通
    # signal = takeBaselineExcursion(signal, 0.55, 0, 128, "highpass")  # 去基线漂移后
    return signal


# 一、3.4 EMGt去干扰      EMG肌电图
def EMGtFilter(signal):
    # signal = takeBaselineExcursion(signal, 0.7, 0, 128, "lowpass")  # 去高频毛刺，低通
    # signal = takeBaselineExcursion(signal, 0.55, 0, 128, "highpass")  # 去基线漂移后
    return signal


# 一、3.2 GSR去干扰    GSR皮肤电反应
def GSRFilter(signal):
    # signal = takeBaselineExcursion(signal, 0.34, 0, 128, "lowpass")  # 去高频毛刺，低通  (信号，频率1，频率2，采样率，方式（高/低/带）)
    # signal = takeBaselineExcursion(signal, 0.58, 0, 128, "highpass")  # 去基线漂移后
    return signal


# 一、3.4 RSP去干扰      呼吸
def RSPFilter(signal):
    # signal = takeBaselineExcursion(signal, 0.9, 0, 128, "lowpass")  # 带通滤波
    # signal = takeBaselineExcursion(signal, 0.5, 0, 128, "highpass")  # 去基线漂移后
    return signal


# 一、3.5 PPG去干扰    PPG 光电脉搏信号
def PPGFilter(signal):
    # signal = takeBaselineExcursion(signal, 0.9, 0, 128, "lowpass")  # 带通滤波
    # signal = takeBaselineExcursion(signal, 0.5, 0, 128, "highpass")  # 去基线漂移后
    return signal


# 一、3.6 SKT去干扰 皮肤温度信号
def SKTFilter(signal):
    # signal = takeBaselineExcursion(signal, 0.8, 0, 128, "lowpass")  # 去高频毛刺，低通
    # signal = takeBaselineExcursion(signal, 0.49, 0, 128, "highpass")  # 去基线漂移后
    return signal