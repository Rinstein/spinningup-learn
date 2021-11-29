"""
@Author: Lrw
@Time: 2021/10/9
功能说明: 绘制训练奖励曲线，对比各个曲线变化，测试算法效果
"""
from matplotlib import pyplot as plt
import numpy as np


def plot_test():
    # path_list = ['./ddpg/HalfCheetah-v2(11, 4, 16, 29, 59).npy', './ppo/HalfCheetah-v2(10, 9, 15, 46, 26).npy', './vpg/HalfCheetah-v2(10, 9, 15, 46, 22).npy']
    # path_list = ['./ddpg/BipedalWalker-v3(11, 4, 19, 22, 35).npy', './ppo/BipedalWalker-v3(11, 4, 17, 22, 51).npy']
    path_list = ['./ppo/BipedalWalker-v3(11, 10, 10, 19, 13).npy', './ddpg/BipedalWalker-v3(11, 10, 15, 49, 32).npy']
    line_name = ['ppo', 'ddpg', 'vpg']
    for index in range(len(path_list)):
        data = np.load(path_list[index])
        data = data[:int(data.shape[0])]
        for i in range(data.shape[0]):
            if i == 0:
                continue
            data[i] = data[i-1]*0.9+data[i]*0.1
        plt.plot(np.arange(1, data.shape[0]+1)*5000, data, label=line_name[index])
        # plt.plot(np.arange(1, data.shape[0]+1), data, label=line_name[index])
    plt.legend()
    plt.show()

def plot_test_t():
    # path_list = ['./ddpg/HalfCheetah-v2(11, 4, 16, 29, 59).npy', './ppo/HalfCheetah-v2(10, 9, 15, 46, 26).npy', './vpg/HalfCheetah-v2(10, 9, 15, 46, 22).npy']
    # path_list = ['./ddpg/BipedalWalker-v3(11, 4, 19, 22, 35).npy', './ppo/BipedalWalker-v3(11, 4, 17, 22, 51).npy']
    path_list = ['./ppo/BipedalWalker-v3(11, 10, 10, 19, 13).npy', './ddpg/BipedalWalker-v3(11, 10, 15, 49, 32).npy']
    line_name = ['COMA', 'IPPO', 'vpg']
    for index in range(len(path_list)):
        data = np.load(path_list[index])
        data = data[:int(data.shape[0])]
        data/=20
        for i in range(data.shape[0]):
            if i == 0:
                continue
            data[i] = data[i-1]*0.9+data[i]*0.1
        plt.plot(np.arange(1, data.shape[0]+1)*10, data, label=line_name[index])
        # plt.plot(np.arange(1, data.shape[0]+1), data, label=line_name[index])
    plt.legend()
    plt.show()

if __name__ == '__main__':
    plot_test()