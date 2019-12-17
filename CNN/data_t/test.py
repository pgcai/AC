# coding:utf-8
import os
import time
import tensorflow as tf
from input_data import readFile, data_reshape_test
from pre_process import pre_data_reshape, more_norm_dataset, more_dataset_1Dto2D
import input_data
import forward
import backward
import numpy as np
import pickle

# 选择GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config = config)

PEOPEL_NUM = input_data.PEOPEL_NUM
test_num_examples = PEOPEL_NUM*40//4  # 测试组数目
TEST_INTERVAL_SECS = 5  # 寻找最新模型等待时间s


def test(signal_re, labels_re):

    # 创建一个默认图,在该图中执行以下操作(多数操作和train中一样)
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [
            test_num_examples,
            forward.IMAGE_SIZE_X,
            forward.IMAGE_SIZE_Y,
            forward.NUM_CHANNELS])
        y_ = tf.placeholder(tf.float32, [None, forward.OUTPUT_NODE])  # ??
        y = forward.forward(x, False, None)  # ??

        ema = tf.train.ExponentialMovingAverage(backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)

        # correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))  # 判断预测值和实际值是否相同
        # correct_prediction = (abs(y-y_) < 0.5)  # 预测结果与真实结果相差是否小于0.5
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 求平均值得到准确率

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(backward.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)

                    # 根据读入的模型名字切分出该模型是属于迭代了多少次保存的
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    reshaped_x = np.reshape(signal_re, (
                        test_num_examples,
                        forward.IMAGE_SIZE_X,
                        forward.IMAGE_SIZE_Y,
                        forward.NUM_CHANNELS))
                    # 计算出测试集上准确率
                    pred_labels = sess.run(y, feed_dict={x: reshaped_x})
                    y_ = labels_re
                    pred_correct = 0
                    for i in range(test_num_examples):
                        print("第{}组数据的预测结果：".format(i))
                        print("预测结果为：{}".format(pred_labels[i]))  # 测试用
                        print("正确结果为：{}".format(y_[i]))
                        j = 0
                        if (y_[i][j] > y_[i][j+1]) == (pred_labels[i][j] > pred_labels[i][j+1]):
                                pred_correct += 1
                        elif (y_[i][j] < y_[i][j+1]) == (pred_labels[i][j] < pred_labels[i][j+1]):
                                pred_correct += 1
                        print("预测正确个数：{}".format(pred_correct))
                        print("当前已测试个数：{}".format(i+1))
                        print("测试数据总个数：{}".format(test_num_examples))
                        print("-------------我是可爱的分割线-----------")  # 测试用
                    Recognition_rate = pred_correct / test_num_examples
                    # pred_labels = np.asarray(reshaped_x)
                    # print(pred_labels)  # 测试用
                    print("----------我是可爱的分割线----------")  # debug用
                    # print(labels_re)  # 测试用
                    print("After {0} training step(s), test accuracy{1}".format(global_step, Recognition_rate))
                else:
                    print('No checkpoint file found')
                    return
            time.sleep(TEST_INTERVAL_SECS)  # 每隔TEST_INTERVAL_SECS秒寻找一次是否有最新模型


def main():
#     signal_data, signal_labels = readFile(backward.filepath)
#     signal_re, labels_re = data_reshape_test(signal_data, signal_labels)
#     print("signal_re.shape:")
#     print(signal_re.shape)
#     print("labels_re.shape:")
#     print(labels_re.shape)
#     re_data = pre_data_reshape(signal_re)
#     print("re_data.shape:")
#     print(re_data.shape)
#     z_score_data = more_norm_dataset(re_data)
#     data_1Dto2D = more_dataset_1Dto2D(z_score_data)
#     print("data_1Dto2D.shape:")
#     print(data_1Dto2D.shape)
#     test(data_1Dto2D, labels_re)
    print("----------开始读入.pkl文件--------")
    with open('/home/superlee/CC/dataset/CNN_test.pkl', 'rb') as f:
        data =  pickle.load(f)
    data_1Dto2D = data['data']
    labels_re = data['labels']
    print("文件已读取！")
    test(data_1Dto2D, labels_re)


if __name__ == '__main__':
    main()
