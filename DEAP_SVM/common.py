import os

# 定义了一些常用的，容易混淆或忘记的方法和功能
import matplotlib.pyplot as plt
# 读取二维列表的某一列【list】
    #正因为列表可以存放不同类型的数据，
    # 因此列表中每个元素的大小可以相同，也可以不同，也就不支持一次性读取一列，
    # 即使是对于标准的二维数字列表
def getrow(thelist, n):
    # 第一个参数为操作的列表，第二个参数为取第几列
    row = [x[n] for x in thelist]
    # print(row)
    return row  # 一列，一维数组

# 画图
# 画图需要提供图片存储路径和图片名称，保存图片
def plot(signal, path, name):
    X = range(len(signal))  # 横坐标，即X轴
    Y = signal  # 竖坐标
    plt.plot(X, Y)

    try:
        if not os.path.exists(path):
            print("路径不存在，已自动建立。")
            os.makedirs(path)
        plt.savefig(path + name)
    except Exception as err:
        print(err)
    plt.close()
    plt.show()

# 写入数据到文件
def writeDataToFile(filepath, filename, data):
    try:
        if not os.path.exists(filepath):
            print("路径不存在，已自动建立。")
            os.makedirs(filepath)       # 在filepath路径下创建文件夹
        #print(data)
        file = open(filepath + filename, 'w')
        file.write(data)
        file.close()
    except Exception as err:
        print(err)

def plotMore(x1, x2, x3, x4, x5):
        fig, ax = plt.subplots(5, 1)

        ax[0].plot(range(len(x1)), x1)
        ax[0].set_xlabel('Time')
        ax[0].set_ylabel('Amplitude')

        ax[1].plot(range(len(x2)), x2, 'r')  # plotting the spectrum
        ax[1].set_xlabel('Freq (Hz)')
        ax[1].set_ylabel('|Y(freq)|')

        ax[2].plot(range(len(x3)), x3, 'G')  # plotting the spectrum
        ax[2].set_xlabel('Freq (Hz)')
        ax[2].set_ylabel('|Y(freq)|')

        ax[3].plot(range(len(x4)), x4, 'B')  # plotting the spectrum
        ax[3].set_xlabel('Freq (Hz)')
        ax[3].set_ylabel('|Y(freq)|')

        ax[4].plot(range(len(x5)), x5, 'B')  # plotting the spectrum
        ax[4].set_xlabel('Freq (Hz)')
        ax[4].set_ylabel('|Y(freq)|')

        plt.show()


def getMean(signal):
    # 求信号均值，参数为一维数组
    sum = 0
    for i in signal:
        sum += i
    avg = sum / len(signal)
    return avg


def getMax(signal):
    # 求最大值
    return max(signal)


def getMin(signal):
    # 求最小值
    return min(signal)

# income.count("99999999")  统计income中“99999999”出现的次数【计算某元素在列表中出现的次数】