将CNN和基于Attention机制的LSTM相结合，利用脑电图地形图和生物信号进行多模态情绪识别。
==========================
Abstract:
---------------------------

### 从生理信号中自动检测情绪状态一直是机器学习算法的一个困难任务。以前关于情绪识别的研究主要集中在精细的手工制作和高度工程化的特征提取上。这些作品的结果证明了区分时空特征对模拟不同情绪的持续演化的重要性。近年来，基于原始信号的图像表征在基于生理的情感识别中取得了良好的效果。如何学习有效的生理情感识别的空间特性，一直是深层表征的基本问题。本文提出了一种高效的多模态识别的深度神经网络框架。首先，利用32通道脑电信号生成脑地形图，获取信号的空间特性。再将脑地形图放入CNN网络提取空间特征。然后将原始生理信号转换为频谱图像，获取信号的时间和频率信息。其次，基于注意力的双向长短时记忆递归神经网络(LSTM-RNNs)被用来自动学习最佳的时间特征。然后将CNN学习到的空间特征和LSTM学习的时序特征输入深度神经网络(DNN)，预测每个通道的情绪输出概率。最后利用决策级融合策略预测最终的情绪。在DEAP和AMIGOS数据集上的实验结果表明，我们的方法优于其他最先进的方法。  

### Keywords-EEG, emotion recognition, deep learning, CNN, LSTM, Attention

## INTRODUCTION
>To be filled

## METHODS
### Pre-processing
>To be filled
### etc.
>To be filled

## EXPERIMENTS
### The Datasets
>To be filled
### Model Implemention
>To be filled
### Results
>To be filled

## CONCLUSION
>To be filled

## REFERENCES
>To be filled

![Fig.1](https://images.cnblogs.com/cnblogs_com/cpg123/1609385/o_191209005159paper_pic01.jpg)
