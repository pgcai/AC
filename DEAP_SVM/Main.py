 # 主类，整体流程框架在这里
# 命名规范：类名 开头大写，分隔用大写字母，不用下划线
# 函数名称 不带下划线，分隔用大写字母，开头小写字母。
# 变量名 全小写，分隔用下划线。
# 信号采样频率128hz

# 调用库及方法
from sklearn import svm
from sklearn import tree
import sys
from common import plot, writeDataToFile
from PyQt5.QtWidgets import QWidget, QPushButton, QApplication, QProgressBar, QLabel, QVBoxLayout, QHBoxLayout
from PyQt5.QtCore import Qt
from tkinter.filedialog import askopenfilename
import os
import gc
import scipy.io as scio
from Filter import takeBaselineExcursion        # 滤波
from FrequencyDomainFeatures import FrequencyDomainFeatures         # 频域特征
from TimeDomainFeatures import TimeDomainFeatures        # 时域特征
from Signal_Preprocessing import EEGFilter, EOGhFilter, EOGvFilter, EMGzFilter, EMGtFilter, GSRFilter, RSPFilter, PPGFilter, SKTFilter  # 滤波

def getPower(signal):
    sums = 0
    for i in range(len(signal)):
        sums = sums + (signal[i] * signal[i])
    sums = int(sums)
    return sums
def getFrequencypower(signal):
    sum = 0
    for i in range(len(signal)):
        sum = sum + (i * signal[i] * signal[i])
    return sum

timeDomainFeatures = TimeDomainFeatures() # 时域特征
frequencyDomainFeatures = FrequencyDomainFeatures()  # 频域特征

class Main(QWidget):
    # 初始化
    def __init__(self):    # 构造函数
        super().__init__()  # 固定写法

        # 初始化界面
        self.initUI()

        self.data = None  # 从文件中读取的数据，并经过预处理，是整个流程的输入
        self.labels = None  # 从文件中读取的标签，用于分类器识别

        self.labelsV = None  # 单独标签，只从V维度看
        self.labelsA = None  # 单独标签，只从A维度看
        self.labelsD = None  # 单独标签，只从D维度看
        self.labelsL = None  # 单独标签，只从L维度看

        # 训练集的位置数组【一维数组】，在三、3中计算识别率
        self.test_posi = []
        self.train_posi = []

        self.file_path = ""  # 文件路径，从图形化界面获取

        self.is_save_file = False  # 是否保存中间数据到文件夹 c-是否保存处理后数据保存到文件夹 ！！！
        # 构建训练器 # 分四维
        # self.clf = svm.SVC(C=2, kernel='linear', decision_function_shape='ovr')
        # C为线性核  kernel 高斯核 ovr 一对多分类 ovo 二分类
        self.clf = tree.DecisionTreeClassifier()
        self.clfV = tree.DecisionTreeClassifier()
        self.clfA = tree.DecisionTreeClassifier()
        self.clfD = tree.DecisionTreeClassifier()
        self.clfL = tree.DecisionTreeClassifier()

    # 图形化界面初始化，绘制所有组件，把不用的隐藏，用到时再显示
    def initUI(self):
        # 创建一个静态文本
        self.Titel = QLabel(self)
        self.Titel.setText("基于多模态生理信号的情绪识别系统")
        self.Titel.setAlignment(Qt.AlignHCenter)

        # 创建一个静态文本
        self.file_path_text = QLabel(self)  # 文件路径名/选择路径提示语
        self.file_path_text.setText("点击右侧按钮，选择生理信号文件")

        # 创建按钮
        self.choose_btn = QPushButton("选择", self)
        self.choose_btn.clicked.connect(self.getPath)  # 绑定函数
        self.choose_btn.setGeometry(0, 0, 50, 25)

        # 创建小布局，装一行
        hbox = QHBoxLayout()
        hbox.addWidget(self.file_path_text, alignment=Qt.AlignCenter)
        hbox.addWidget(self.choose_btn, alignment=Qt.AlignRight)

        # 开始按钮
        self.start_btn = QPushButton("开始", self)
        self.start_btn.clicked.connect(self.step)
        self.start_btn.setGeometry(0, 0, 50, 25)
        self.start_btn.setVisible(False)

        # 进度条：
        self.classify_bar = QProgressBar(self)
        self.classify_bar.setVisible(False)
        self.filter_bar = QProgressBar(self)
        self.filter_bar.setVisible(False)
        self.feat_extract_bar = QProgressBar(self)
        self.feat_extract_bar.setVisible(False)
        self.recognition_bar = QProgressBar(self)
        self.recognition_bar.setVisible(False)

        # 进度条说明文字：
        self.classify_bar_text = QLabel(self)
        self.classify_bar_text.setText("将信号分类：")
        self.classify_bar_text.setVisible(False)
        self.filter_bar_text = QLabel(self)
        self.filter_bar_text.setText("去除噪声：")
        self.filter_bar_text.setVisible(False)
        self.feat_extract_bar_text = QLabel(self)
        self.feat_extract_bar_text.setText("提取特征：")
        self.feat_extract_bar_text.setVisible(False)
        self.recognition_bar_text = QLabel(self)
        self.recognition_bar_text.setText("识别：")
        self.recognition_bar_text.setVisible(False)


        # 点击查看按钮
        self.filter_btn = QPushButton("点击查看处理后的信号", self)
        self.filter_btn.clicked.connect(self.openFile_Filter)
        self.filter_btn.setVisible(False)
        self.feat_btn = QPushButton("点击查看信号的特征", self)
        self.feat_btn.clicked.connect(self.openFile_Feat)
        self.feat_btn.setVisible(False)
        self.recognition_btn = QPushButton("点击查看识别报告", self)
        self.recognition_btn.clicked.connect(self.openFile)
        self.recognition_btn.setVisible(False)


        # 垂直布局相关属性设置
        vbox = QVBoxLayout()
        # 添加标签到垂直布局中
        vbox.addWidget(self.Titel)
        # 鼠标垂直拉伸不会改变高度
        #vbox.addStretch()
        vbox.addLayout(hbox)
        vbox.addStretch()
        vbox.addWidget(self.start_btn, alignment=Qt.AlignCenter)
        vbox.addWidget(self.classify_bar_text)
        vbox.addWidget(self.classify_bar)
        vbox.addWidget(self.filter_bar_text)
        vbox.addWidget(self.filter_bar)
        vbox.addWidget(self.filter_btn)
        vbox.addWidget(self.feat_extract_bar_text)
        vbox.addWidget(self.feat_extract_bar)
        vbox.addWidget(self.feat_btn)
        vbox.addWidget(self.recognition_bar_text)
        vbox.addWidget(self.recognition_bar)
        vbox.addWidget(self.recognition_btn)



        # 加载布局：前面设置好的垂直布局
        self.setLayout(vbox)
        self.setGeometry(300, 300, 400, 500)
        self.setWindowTitle('Tooltips')  # 设置标题
        self.show()

    # 打开文件夹(总文件夹)
    def openFile(self):
        os.system("start explorer E:\\result\\report")
    # 打开处理后信号文件夹(signalclear)
    def openFile_Filter(self):
        os.system("start explorer E:\\result\\signalclear")
    # 打开提取特征文件夹(feat)
    def openFile_Feat(self):
        os.system("start explorer E:\\result\\feat")
    # 通过图形化界面选择文件路径
    def getPath(self):
        # 选择文件夹(返回一个路径字符串)
        path = askopenfilename()
        self.file_path = path
        self.start_btn.setVisible(True)


    # 各信号特征提取 函数

    # 二、1.1 提取EEG信号的特征:
    def getEEGFeat(self, signal):
        # 输入：一维数组  输出：一维数组+说明文字打印
        EEGfeat = []
        FFT = frequencyDomainFeatures.FFT(signal)  # 频谱图
        pp = timeDomainFeatures.getPeak_to_peak(signal)  # 峰峰值
        sd = timeDomainFeatures.getStandardDeviation(signal)  # 标准差

        filter2 = takeBaselineExcursion(signal, 14, 31, 128, "bandpass")[int(len(signal) * 2 / 15): int(len(signal) * 11 / 15)]
        power2 = getPower(filter2)  # β：14-30
        power4 = getFrequencypower(FFT[14:30])
        EEGfeat.extend([pp, sd, power2, power4])
        return EEGfeat

    # 二、1.2 提取EOGh信号的特征:
    def getEOGhFeat(self, signal):
        # 输入：一维数组  输出：一维数组+说明文字打印
        EOGfeat = []
        sd = (timeDomainFeatures.getStandardDeviation(signal))  # 标准差
        mean = timeDomainFeatures.getMean(signal)  # 均值
        EOGfeat.extend([sd, mean])
        return EOGfeat

    # 二、1.3 提取EOGv信号的特征:
    def getEOGvFeat(self, signal):
        # 输入：一维数组  输出：一维数组+说明文字打印
        EOGfeat = []
        sd = (timeDomainFeatures.getStandardDeviation(signal))  # 标准差
        mean = timeDomainFeatures.getMean(signal)  # 均值
        EOGfeat.extend([sd, mean])
        return EOGfeat

    # 二、1.4 提取EMGz信号的特征:
    def getEMGzFeat(self, signal):
        # 输入：一维数组  输出：一维数组+说明文字打印
        EMGfeat = []
        sd = (timeDomainFeatures.getStandardDeviation(signal))  # 标准差
        mean = timeDomainFeatures.getMean(signal)  # 均值
        EMGfeat.extend([sd, mean])
        return EMGfeat

    # 二、1.5 提取EMGt信号的特征:
    def getEMGtFeat(self, signal):
        # 输入：一维数组  输出：一维数组+说明文字打印
        EMGfeat = []
        sd = (timeDomainFeatures.getStandardDeviation(signal))  # 标准差
        mean = timeDomainFeatures.getMean(signal)  # 均值
        EMGfeat.extend([sd, mean])
        return EMGfeat

    # 二、1.6 提取GSR信号的特征
    def getGSRFeat(self, signal):
        # 输入：一维数组  输出：一维数组+说明文字打印
        GSRfeat = []
        mean = timeDomainFeatures.getMean(signal)  # 均值
        sd = (timeDomainFeatures.getStandardDeviation(signal))  # 标准差
        GSRfeat.extend([sd, mean])
        return GSRfeat

    # 二、1.7 提取RSP信号的特征:
    def getRSPFeat(self, signal):
        # 输入：一维数组  输出：一维数组+说明文字打印
        PPGfeat = []
        variance = timeDomainFeatures.getVariance(signal)  # 方差
        mean = timeDomainFeatures.getMean(signal)  # 均值
        PPGfeat.extend([variance, mean])
        return PPGfeat

    # 二、1.8 提取PPG信号的特征
    def getPPGFeat(self, signal):
        # 输入：一维数组  输出：一维数组+说明文字打印
        PPGfeat = []
        variance = timeDomainFeatures.getVariance(signal)  # 方差
        mean = timeDomainFeatures.getMean(signal)  # 均值
        PPGfeat.extend([variance, mean])
        return PPGfeat

    # 二、1.9 提取SKT信号的特征
    def getSKTFeat(self, signal):
        # 输入：一维数组  输出：一维数组+说明文字打印
        SKTfeat = []
        sd = (timeDomainFeatures.getStandardDeviation(signal))  # 标准差
        mean = timeDomainFeatures.getMean(signal)  # 均值
        SKTfeat.extend([sd, mean])
        return SKTfeat

    # 三、2.1 将labels转为一维数组，并分为四小类
    def transLabels(self):
        # self.labels转一维数组
        # print(len(self.labels))  # 测试用
        L1 = [0] * len(self.labels)
        # 完整版
        for j in range(len(self.labels)):  # j: 1*4 数组
            try:
                if self.labels[j][0] >= 5:  # HV
                    if self.labels[j][1] >= 5:  # HA
                        if self.labels[j][2] >= 5:  # HD
                            if self.labels[j][3] >= 5:  # HL
                                L1[j] = 0
                            elif self.labels[j][3] < 5:  # LL
                                L1[j] = 1
                        elif self.labels[j][2] < 5:  # LD
                            if self.labels[j][3] >= 5:  # HL
                                L1[j] = 2
                            elif self.labels[j][3] < 5:  # LL
                                L1[j] = 3
                    elif self.labels[j][1] < 5:  # LA
                        if self.labels[j][2] >= 5:  # HD
                            if self.labels[j][3] >= 5:  # HL
                                L1[j] = 4
                            elif self.labels[j][3] < 5:  # LL
                                L1[j] = 5
                        elif self.labels[j][2] < 5:  # LD
                            if self.labels[j][3] >= 5:  # HL
                                L1[j] = 6
                            elif self.labels[j][3] < 5:  # LL
                                L1[j] = 7
                elif self.labels[j][0] < 5:  # LV
                    if self.labels[j][1] >= 5:  # HA
                        if self.labels[j][2] >= 5:  # HD
                            if self.labels[j][3] >= 5:  # HL
                                L1[j] = 8
                            elif self.labels[j][3] < 5:  # LL
                                L1[j] = 9
                        elif self.labels[j][2] < 5:  # LD
                            if self.labels[j][3] >= 5:  # HL
                                L1[j] = 10
                            elif self.labels[j][3] < 5:  # LL
                                L1[j] = 11
                    elif self.labels[j][1] < 5:  # LA
                        if self.labels[j][2] >= 5:  # HD
                            if self.labels[j][3] >= 5:  # HL
                                L1[j] = 12
                            elif self.labels[j][3] < 5:  # LL
                                L1[j] = 13
                        elif self.labels[j][2] < 5:  # LD
                            if self.labels[j][3] >= 5:  # HL
                                L1[j] = 14
                            elif self.labels[j][3] < 5:  # LL
                                L1[j] = 15
            except Exception as err:
                print(err)
            self.recognition_bar.setValue(50 + (j + 1.0) / len(self.labels) * 50)    # 进度条

        # V 版
        v = [0] * len(self.labels)
        for i in range(len(self.labels)):
            if self.labels[i][0] > 5:
                v[i] = 1
            elif self.labels[i][0] <= 5:
                v[i] = 2
        self.labelsV = v

        # A 版
        a = [0] * len(self.labels)
        for i in range(len(self.labels)):
            if self.labels[i][1] > 5:
                a[i] = 1
            elif self.labels[i][1] <= 5:
                a[i] = 2
        self.labelsA = a

        # D 版
        d = [0] * len(self.labels)
        for i in range(len(self.labels)):
            if self.labels[i][2] > 5:
                d[i] = 1
            elif self.labels[i][2] <= 5:
                d[i] = 2
        self.labelsD = d

        # L 版
        l = [0] * len(self.labels)
        # print(self.labels)
        for i in range(len(self.labels)):
            # print(self.labels[i][3]+"  ")
            if self.labels[i][3] > 5:
                l[i] = 1
            elif self.labels[i][3] <= 5:
                l[i] = 2
        self.labelsL = l

        self.labels = L1

    # 一、1.读取.csv文件
    def readFile(self, file_path):
        data = scio.loadmat(file_path)  # 读取mat文件数据
        signal_data = data['data'].tolist()  # data数据 将数据以列表形式作为值，data作为键 的字典格式传递给signal_data
        signal_labels = data['labels'].tolist()  # labels数据 将标签数据以列表形式作为值，labels作为键 的字典格式传递给signal_labels
        # ---------

        # 手动释放内存
        del data
        gc.collect()
        return signal_data, signal_labels  # 包括data（40*40*8064）和labels（40*4）

    # 一、2. 按信号种类分类存储
    def signalClassify(self, data):
        self.classify_bar_text.setVisible(True)
        self.classify_bar.setVisible(True)
        self.classify_bar.setValue(0)
        # ----------------------以上code为图形化界面相关，以下code为算法相关
        data_dict_list = [0] * 40
        for j in range(40):  # 共40组
            self.classify_bar.setValue(100.0 / 40 * (j+1))  # 进度条
            data_dict_list[j] = dict()  # 创建空字典
            # 脑电  EEG0 EEG1 ... EEG31
            for i in range(32):  # 前32组为脑电
                EEG = []
                EEG.extend(data[j][i])
                data_dict_list[j]["EEG"+str(i)] = EEG

            # 眼动信号EOG信号 32：EOGh 33:EOGv
            EOGh = []
            EOGh.extend(data[j][32])
            data_dict_list[j]['EOGh'] = EOGh

            EOGv = []
            EOGv.extend(data[j][33])
            data_dict_list[j]['EOGv'] = EOGv

            # 肌电信号  34：EMGz 颧肌 35：EMGt 斜方肌
            EMGz = []
            EMGz.extend(data[j][34])
            data_dict_list[j]['EMGz'] = EMGz

            EMGt = []
            EMGt.extend(data[j][35])
            data_dict_list[j]['EMGt'] = EMGt

            # 皮肤电信号  GSR 36
            GSR = []
            GSR.extend(data[j][36])
            data_dict_list[j]['GSR'] = GSR

            # 呼吸 RSP  37
            RSP = []
            RSP.extend(data[j][37])
            data_dict_list[j]['RSP'] = RSP

            # 光电脉搏信号 PPG 38
            PPG = []
            PPG.extend(data[j][38])
            data_dict_list[j]['PPG'] = PPG

            # 温度 SKT 39
            SKT = []
            SKT.extend(data[j][39])
            data_dict_list[j]['SKT'] = SKT

        data_dict = data_dict_list
        return data_dict  # 返回：数据字典数组

    # 二、1.对不同种类信号，进行预处理(去除毛刺、噪声、基线漂移等干扰)
    def signalFilter(self, data_dict):
        self.filter_bar_text.setVisible(True)
        self.filter_bar.setVisible(True)
        self.filter_bar.setValue(0)
        self.filter_btn.setVisible(True)
        # ----------------------以上为图形化界面相关，以下为算法相关

        for j in range(len(data_dict)):  # 数据字典数组中共40个字典
            # 脑电
            for i in range(32):
                EEG = data_dict[j]["EEG"+str(i)]  # 从字典中还原出list
                EEG = EEGFilter(EEG)
                data_dict[j]["EEG"+str(i)] = EEG
                if self.is_save_file:
                    plot(EEG, r"E:\result\Signalclear\EEG\\", "filter_EEG_" + str(j)+"_"+str(i))

            # 眼动信号EOG信号 32：EOGh 33:EOGv
            EOGh = data_dict[j]['EOGh']
            EOGh = EOGhFilter(EOGh)
            data_dict[j]['EOGh'] = EOGh
            if self.is_save_file:
                plot(EOGh, r"E:\result\Signalclear\EOGh\\", "filter_EOGh_" + str(j))

            EOGv = data_dict[j]['EOGv']
            EOGv = EOGvFilter(EOGv)
            data_dict[j]['EOGv'] = EOGv
            if self.is_save_file:
                plot(EOGv, r"E:\result\Signalclear\EOGv\\", "filter_EOGv_" + str(j))

            # 肌电信号  34：EMGz 颧肌 35：EMGt 斜方肌
            EMGz = data_dict[j]['EMGz']
            EMGz = EMGzFilter(EMGz)
            data_dict[j]['EMGz'] = EMGz
            if self.is_save_file:
                plot(EMGz, r"E:\result\Signalclear\EMGz\\", "filter_EMGz_" + str(j))

            EMGt = data_dict[j]['EMGt']
            EMGt = EMGtFilter(EMGt)
            data_dict[j]['EMGt'] = EMGt
            if self.is_save_file:
                plot(EMGt, r"E:\result\Signalclear\EMGt\\", "filter_EMGt_" + str(j))

            # 皮肤电信号  GSR 36
            GSR = data_dict[j]['GSR']
            GSR = GSRFilter(GSR)
            data_dict[j]['GSR'] = GSR
            if self.is_save_file:
                plot(GSR, r"E:\result\Signalclear\GSR\\", "filter_GSR_" + str(j))

            # 呼吸 RSP
            RSP = data_dict[j]['RSP']
            RSP = RSPFilter(RSP)
            data_dict[j]['RSP'] = RSP
            if self.is_save_file:
                plot(RSP, r"E:\result\Signalclear\RSP\\", "filter_RSP_" + str(j))

            # 光电脉搏 PPG
            PPG = data_dict[j]['PPG']
            PPG = PPGFilter(PPG)
            data_dict[j]['PPG'] = PPG
            if self.is_save_file:
                plot(PPG, r"E:\result\Signalclear\PPG\\", "filter_PPG_" + str(j))

            # 皮温
            SKT = data_dict[j]['SKT']
            SKT = SKTFilter(SKT)
            data_dict[j]['SKT'] = SKT
            if self.is_save_file:
                plot(SKT, r"E:\result\Signalclear\SKT\\", "filter_SKT_" + str(j))

            self.filter_bar.setValue((j+1.0) / len(data_dict) * 100)  # 进度条

        data_dict_clear = data_dict
        return data_dict_clear  # 返回：数据字典数组

    # 三、1.对不同种类信号，进行特征提取
    def getFeatDict(self, data_dict_clear):
        self.feat_extract_bar_text.setVisible(True)
        self.feat_extract_bar.setVisible(True)
        self.feat_extract_bar.setValue(0)
        # ----------------------以上为图形化界面相关，以下为算法相关

        for j in range(len(data_dict_clear)):  # 40组数据字典

            # 脑电
            for i in range(5):
                EEG = data_dict_clear[j]['EEG'+str(i)]  # 从字典中还原出list
                EEG = self.getEEGFeat(EEG)
                data_dict_clear[j]['EEG'+str(i)] = EEG
                try:
                    data = ""
                    for k in range(len(EEG)):
                        data = data.join(str(EEG[k]))
                        data = data.join("    ")
                    writeDataToFile(r"E:\result\feat\EEG\\", "EEG" + str(k)+".txt", data)
                except Exception as err:
                    print(err)

            # 眼动信号EOG信号 32：EOGh 33:EOGv
            EOGh = data_dict_clear[j]['EOGh']
            EOGh = self.getEOGhFeat(EOGh)
            data_dict_clear[j]['EOGh'] = EOGh
            try:
                data = ""
                for i in range(len(EOGh)):
                    data = data.join(str(EOGh[i]))
                    data = data.join("    ")
                writeDataToFile(r"E:\result\feat\EOGh\\", "EOGh" + str(j)+".txt", data)
            except Exception as err:
                print(err)

            EOGv = data_dict_clear[j]['EOGv']
            EOGv = self.getEOGvFeat(EOGv)
            data_dict_clear[j]['EOGv'] = EOGv
            try:
                data = ""
                for i in range(len(EOGv)):
                    data = data.join(str(EOGv[i]))
                    data = data.join("    ")
                writeDataToFile(r"E:\result\feat\EOGv\\", "EOGv" + str(j) + ".txt", data)
            except Exception as err:
                print(err)

            # 肌电信号  34：EMGz 颧肌 35：EMGt 斜方肌
            EMGz = data_dict_clear[j]['EMGz']
            EMGz = self.getEMGzFeat(EMGz)
            data_dict_clear[j]['EMGz'] = EMGz
            try:
                data = ""
                for i in range(len(EMGz)):
                    data = data.join(str(EMGz[i]))
                    data = data.join("    ")
                writeDataToFile(r"E:\result\feat\EMGz\\", "EMGz" + str(j)+".txt", data)
            except Exception as err:
                print(err)

            EMGt = data_dict_clear[j]['EMGt']
            EMGt = self.getEMGtFeat(EMGt)
            data_dict_clear[j]['EMGt'] = EMGt
            try:
                data = ""
                for i in range(len(EMGt)):
                    data = data.join(str(EMGt[i]))
                    data = data.join("    ")
                writeDataToFile(r"E:\result\feat\EMGt\\", "EMGt" + str(j) + ".txt", data)
            except Exception as err:
                print(err)

            # 皮肤电GSR
            GSR = data_dict_clear[j]['GSR']
            GSR = self.getGSRFeat(GSR)
            data_dict_clear[j]['GSR'] = GSR
            try:
                data = ""
                for i in range(len(GSR)):
                    data = data.join(str(GSR[i]))
                    data = data.join("    ")
                writeDataToFile(r"E:\result\feat\GSR\\", "GSR" + str(j)+".txt", data)
            except Exception as err:
                print(err)

            # 呼吸 RSP
            RSP = data_dict_clear[j]['RSP']
            RSP = self.getRSPFeat(RSP)
            data_dict_clear[j]['RSP'] = RSP
            try:
                data = ""
                for i in range(len(RSP)):
                    data = data.join(str(RSP[i]))
                    data = data.join("    ")
                writeDataToFile(r"E:\result\feat\RSP\\", "RSP" + str(j) + ".txt", data)
            except Exception as err:
                print(err)

            # 光电脉搏
            PPG = data_dict_clear[j]['PPG']
            PPG = self.getPPGFeat(PPG)
            data_dict_clear[j]['PPG'] = PPG
            try:
                data = ""
                for i in range(len(PPG)):
                    data = data.join(str(PPG[i]))
                    data = data.join("    ")
                writeDataToFile(r"E:\result\feat\PPG\\", "PPG" + str(j)+".txt", data)
            except Exception as err:
                print(err)

            # 皮温
            SKT = data_dict_clear[j]['SKT']
            SKT = self.getSKTFeat(SKT)
            data_dict_clear[j]['SKT'] = SKT
            try:
                data = ""
                for i in range(len(SKT)):
                    data = data.join(str(SKT[i]))
                    data = data.join("    ")
                writeDataToFile(r"E:\result\feat\SKT\\", "SKT" + str(j)+".txt", data)
            except Exception as err:
                print(err)

            self.feat_extract_bar.setValue((j + 1.0) / len(data_dict_clear) * 100)  # 进度条

        feat_dict = data_dict_clear
        self.feat_btn.setVisible(True)
        return feat_dict  # 返回：特征字典数组

    # 四、1. 划分训练集与测试集
    def partitionTrainTest(self, feat_dict):
        train_dict = feat_dict[0:20]
        test_dict = feat_dict[20:40]
        # 将测试集的位置信息存储, 在三、3中计算识别率
        self.train_posi = range(0, 20)
        self.test_posi = range(20, 40)
        return train_dict, test_dict  # 返回：训练集字典数组，测试集字典数组

    # 四、2. 使用训练集训练分类器
    def trainClassifier(self, train_dict):
        # 测试集X为二维数组(train_dict转化后)，Y为一维数组(self.labels转化后)
        self.recognition_bar_text.setVisible(True)
        self.recognition_bar.setVisible(True)
        self.recognition_bar.setValue(0)
        # ----------------------以上为图形化界面相关，以下为算法相关

        # train_dict【字典数组】转【二维数组】
        X = [[0]] * len(train_dict)
        for i in range(len(X)):  # 字典转一维数组
            X[i] = []  # 解决奇怪问题，有用的
            for j in range(32):
                X[i].extend(train_dict[i]['EEG'+str(j)])
            X[i].extend(train_dict[i]['EOGh'])
            X[i].extend(train_dict[i]['EOGv'])
            X[i].extend(train_dict[i]['GSR'])
            X[i].extend(train_dict[i]['RSP'])
            X[i].extend(train_dict[i]['PPG'])
            X[i].extend(train_dict[i]['SKT'])
            # print(str(i)+" ")  # 测试用
            # print(X[i])  # 测试用
            self.recognition_bar.setValue((i+1.0) / len(train_dict) * 50)  # 进度条

        self.transLabels()  # 三、2.1 将labels转为一维数组，并分为四小类

        Y  = []
        YV = []
        YA = []
        YD = []
        YL = []
        for i in range(len(X)):  # 根据X的个数，确定Y的个数
            Y.append(self.labels[self.train_posi[i]])
            YV.append(self.labelsV[self.train_posi[i]])
            YA.append(self.labelsA[self.train_posi[i]])
            YD.append(self.labelsD[self.train_posi[i]])
            YL.append(self.labelsL[self.train_posi[i]])

        # 训练
        self.clf.fit(X, Y)
        self.clfV.fit(X, YV)
        self.clfA.fit(X, YA)
        self.clfD.fit(X, YD)
        self.clfL.fit(X, YL)
        pass  # 输出：（全局）分类器

    # 四、3.使用分类器测试识别率
    def classifyData(self, test_dict):
        # test_dict【字典数组】转【二维数组】
        X = [[0]] * len(test_dict)
        for i in range(len(test_dict)):  # 字典转一维数组
            X[i] = []
            for j in range(32):
                X[i].extend(test_dict[i]['EEG' + str(j)])
            X[i].extend(test_dict[i]['EOGh'])
            X[i].extend(test_dict[i]['EOGv'])
            X[i].extend(test_dict[i]['GSR'])
            X[i].extend(test_dict[i]['RSP'])
            X[i].extend(test_dict[i]['PPG'])
            X[i].extend(test_dict[i]['SKT'])

        predict_re = self.clf.predict(X)  # 结果为一维数组
        predict_reV = self.clfV.predict(X)  # 结果为一维数组
        predict_reA = self.clfA.predict(X)  # 结果为一维数组
        predict_reD = self.clfD.predict(X)  # 结果为一维数组
        predict_reL = self.clfL.predict(X)  # 结果为一维数组

        recognition_Right = 0
        recognition_RightV = 0
        recognition_RightA = 0
        recognition_RightD = 0
        recognition_RightL = 0

        for i in range(len(predict_re)):
            if predict_re[i] == self.labels[self.test_posi[i]]:
                recognition_Right += 1
            if predict_reV[i] == self.labelsV[self.test_posi[i]]:
                recognition_RightV += 1
            if predict_reA[i] == self.labelsA[self.test_posi[i]]:
                recognition_RightA += 1
            if predict_reD[i] == self.labelsD[self.test_posi[i]]:
                recognition_RightD += 1
            if predict_reL[i] == self.labelsL[self.test_posi[i]]:
                recognition_RightL += 1
        recognition_rate = recognition_Right * 1.0 / len(test_dict)
        recognition_rateV = recognition_RightV * 1.0 / len(test_dict)
        recognition_rateA = recognition_RightA * 1.0 / len(test_dict)
        recognition_rateD = recognition_RightD * 1.0 / len(test_dict)
        recognition_rateL = recognition_RightL * 1.0 / len(test_dict)

        print("识别率: " + str(recognition_rate) + "  V: " + str(recognition_rateV)+
              " A: " + str(recognition_rateA) + " D: " + str(recognition_rateD) + " L: " + str(recognition_rateL))
        # print("识别结果：" + " All: " + str(recognition_rate) + " V: " + str(recognition_rateV )
        #       + " A: " + str(recognition_rateA) + " D: " + str(recognition_rateD) + "L:" + str(recognition_rateL))
        data = "识别率: " + str(recognition_rate) + "  V: " + str(recognition_rateV)+ \
              " A: " + str(recognition_rateA) + " D: " + str(recognition_rateD) + " L: " + str(recognition_rateL)\
               # +"\n" + "识别结果：" + " All: " + str(recognition_rate) + " V: " + str(recognition_rateV ) + " A: " + \
               # str(recognition_rateA) + " D: " + str(recognition_rateD) + "L:" + str(recognition_rateL)

        writeDataToFile(r"E:\result\report\\", "report.txt", data)
        return recognition_rate  # 返回：识别结果，识别率

    # 总框架
    # 一 二、读取数据及预处理
    def readDataPreprocessing(self):
        self.data, self.labels = self.readFile(self.file_path)  # 1.读取.csv文件
        data_dict = self.signalClassify(self.data)  # 2. 按信号种类分类存储
        data_dict_clear = self.signalFilter(data_dict)  # 3. 针对不同种类信号，去除毛刺、噪声、基线漂移等干扰

        del data_dict
        gc.collect()
        return data_dict_clear  # 输出：数据字典数组

    # 三、特征提取
    def featureExtraction(self, data_dict_clear):
        feat_dict = self.getFeatDict(data_dict_clear)  # 1.按信号种类提取各自的重要特征
        return feat_dict

    # 四、训练分类器及识别
    def classifierAndRecognition(self, feat_dict):
        train_dict, test_dict = self.partitionTrainTest(feat_dict)  # 1.划分训练集与测试集
        self.trainClassifier(train_dict)  # 2. 使用训练集训练分类器
        recognition_rate = self.classifyData(test_dict)  # 3.使用分类器测试识别率
        self.recognition_btn.setVisible(True)  # 显示按钮
        return recognition_rate

    # 流程的主程序
    def step(self):
        print("go!")
        data_dict_clear = self.readDataPreprocessing()  # 一
        feat_dict = self.featureExtraction(data_dict_clear)  # 二
        recognition_rate = self.classifierAndRecognition(feat_dict)  # 三
        # print(recognition_rate)

    # 软件的主程序
    def main(self):
        pass


# 主函数，只负责调出图形化界面，流程工作由按钮调控
if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = Main()
    ex.main()
    sys.exit(app.exec_())
