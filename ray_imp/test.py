import torch
import gym
import numpy as np
from memory_profiler import profile
from matplotlib import pyplot as plt
from ray_imp.common import OnPolicyBuffer


@profile
def my_func():
    a = [1] * (10 ** 6)
    b = [2] * (2 * 10 ** 7)
    del b
    return a


def test_env(env_name):
    env = gym.make(env_name)
    step = 0
    env.reset()
    total_episode = 0
    while True:
        total_episode += 1
        while True:
            env.render()
            step += 1
            obs, r, done, _ = env.step(env.action_space.sample())
            print(step, r, obs)
            if done:
                # env.reset()
                break
        print('total episode', total_episode)

if __name__ == '__main__':
    """内存监控工具"""
    # buffer = OnPolicyBuffer()
    # for i in range(10):
    #     count = 5000
    #     num = 10
    #     for i in range(count):
    #         buffer.add(np.array([1.1]*num, dtype=np.float64), torch.tensor([1.1]*num, dtype=torch.float64), [1.1]*num, [1.1]*num, [1.1]*num, [1.1]*num)
    #     buffer.clear_buffer()
    """数据存储测试"""
    # a = np.array([1] * 100)
    # print(a[np.ones(10, dtype=np.int32)])
    # test_env('HalfCheetah-v2')
    test_env('BipedalWalker-v3')
    # test_env('CartPole-v0')