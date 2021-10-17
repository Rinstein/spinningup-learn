"""
@Author: Lrw
@Time: 2021/10/9
功能说明: 绘制训练奖励曲线，对比各个曲线变化，测试算法效果
"""
from matplotlib import pyplot as plt
import numpy as np


def plot_test():
    path_list = ['./ppo/HalfCheetah-v2(10, 9, 15, 46, 26).npy', './vpg/HalfCheetah-v2(10, 9, 15, 46, 22).npy']
    line_name = ['ppo', 'vpg']
    for index in range(len(path_list)):
        data = np.load(path_list[index])
        plt.plot(np.arange(1, data.shape[0]+1)*5000, data, label=line_name[index])
    plt.legend()
    plt.show()

if __name__ == '__main__':
    plot_test()