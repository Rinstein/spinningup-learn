# spinningup-learn
to make a better understand for RL core concepts through OpenAI SpinningUp project
# todo list
### 一、PPO的优化
1. （基本完成）内存泄漏问题，为何内存占用在一直增长：解决了绝大部分内存泄露问题，模型运行后，不需要计算梯度的时候，最好用torch.no_grad模式启动，并且尽量不要输出tensor变量到其他地方进行计算
2. （已完成，有效提高了数据采样率）k_epoch早停机制，在当前策略和采样策略相差较大时，停止使用数据
3. 标准的ppo还会把entropy加入loss当中
4. 使用GAE优势函数进行测试
5. 既然AC结构中的Critic是基于值的方法，那么是否可以使用类似于DQN的replay buffer来加速Critic模型的训练 
6. The Spinning Up implementation of PPO supports parallelization with MPI, 如何使用MPI接口实现并行化
7. PPO的性能还没有跟上论文里面的效果，而且远低于DDPG
### 二、ddpg算法的实现
1. 已经实现ddpg算法，可以在跳跳狗环境中训练好，但是在2D行走环境（直立行走）中波动性较大
### 三、当前存在的问题
1. PPO可以应用到离散和连续空间中
2. ddpg只有应用到连续空间中
在单智能体下如何解决既有连续空间又有离散空间的决策
### debug RL 算法
1. 如果是loss非常大，奖励波动很大，通常是网络的计算过程出了问题，该增加或减少维度的时候没有做，可以调试一遍整个运行loop，看看各个中间变量的维度是否正确
2. 如果监控的训练变量出现了奇怪的问题，例如loss极大地上升，奖励波动极大，尽可能快速去追踪错误，才能发现问题
