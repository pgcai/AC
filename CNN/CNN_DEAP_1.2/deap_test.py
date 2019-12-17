# coding:utf-8
import time
import tensorflow as tf
from input_data import readFile, data_reshape_test
import deap_forward
import deap_backward
import numpy as np

test_num_examples = 10  # 测试组数目
TEST_INTERVAL_SECS = 5  # 寻找最新模型等待时间s


def test(signal_re, labels_re):

    # 创建一个默认图,在该图中执行以下操作(多数操作和train中一样)
    with tf.Graph().as_default() as g: 
        x = tf.placeholder(tf.float32, [
            test_num_examples,
            deap_forward.IMAGE_SIZE_X,
            deap_forward.IMAGE_SIZE_Y,
            deap_forward.NUM_CHANNELS]) 
        y_ = tf.placeholder(tf.float32, [None, deap_forward.OUTPUT_NODE])  # ??
        y = deap_forward.forward(x, False, None)  # ??

        ema = tf.train.ExponentialMovingAverage(deap_backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)

        # correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))  # 判断预测值和实际值是否相同
        # correct_prediction = (abs(y-y_) < 0.5)  # 预测结果与真实结果相差是否小于0.5
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 求平均值得到准确率

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(deap_backward.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)

                    # 根据读入的模型名字切分出该模型是属于迭代了多少次保存的
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1] 
                    reshaped_x = np.reshape(signal_re, (
                        test_num_examples,
                        deap_forward.IMAGE_SIZE_X,
                        deap_forward.IMAGE_SIZE_Y,
                        deap_forward.NUM_CHANNELS))
                    # 计算出测试集上准确率
                    pred_labels = sess.run(y, feed_dict={x: reshaped_x})
                    y_ = labels_re
                    pred_correct_V = 0
                    pred_correct_A = 0
                    pred_correct_D = 0
                    pred_correct_L = 0
                    for i in range(test_num_examples):
                        print("-------------我是可爱的分割线-----------")  # 测试用
                        for j in range(4):
                            print(pred_labels[i][j])  # 测试用
                            print(y_[i][j])  # 测试用
                            print("-----------")
                            if pred_labels[i][j] >= 0.5:
                                pred_labels[i][j] = 1
                            else:
                                pred_labels[i][j] = 0
                            if pred_labels[i][j] == y_[i][j] and j == 0:
                                pred_correct_V += 1
                            if pred_labels[i][j] == y_[i][j] and j == 1:
                                pred_correct_A += 1
                            if pred_labels[i][j] == y_[i][j] and j == 2:
                                pred_correct_D += 1
                            if pred_labels[i][j] == y_[i][j] and j == 3:
                                pred_correct_L += 1
                    Recognition_rate_V = pred_correct_V/test_num_examples
                    Recognition_rate_A = pred_correct_A / test_num_examples
                    Recognition_rate_D = pred_correct_D / test_num_examples
                    Recognition_rate_L = pred_correct_L / test_num_examples
                    # pred_labels = np.asarray(reshaped_x)
                    # print(pred_labels)  # 测试用
                    print("----------我是可爱的分割线----------")  # debug用
                    # print(labels_re)  # 测试用
                    print("After {0} training step(s), test accuracy V = {1} ,test accuracy A = {2}, "
                          "test accuracy D = {3} ,test accuracy L = {4}".format(global_step, Recognition_rate_V,
                                                                          Recognition_rate_A,
                                                                          Recognition_rate_D,
                                                                          Recognition_rate_L))
                else:
                    print('No checkpoint file found')
                    return
            time.sleep(TEST_INTERVAL_SECS)  # 每隔TEST_INTERVAL_SECS秒寻找一次是否有最新模型


def main():
    signal_data, signal_labels = readFile('F:/情感计算/数据集/DEAP/s02.mat')
    signal_re, labels_re = data_reshape_test(signal_data, signal_labels)
    test(signal_re, labels_re)


if __name__ == '__main__':
    main()
