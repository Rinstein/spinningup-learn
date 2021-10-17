"""
@Author: Lrw
@Time: 2021/9/30
功能说明: 实现ppo clip版本的算法
"""
import time
import argparse
import torch
import numpy as np
from ray_imp.common import *
from torch.optim import Adam
from spinup.utils.logx import EpochLogger


class PPO(rl_algorithm):
    def __init__(self, args, env: gym.Env):
        self.env = env
        obs_dim = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space
        self.args = args
        self.ac = MlpActorCritic(obs_dim, self.args.hidden_sizes, self.action_space, nn.Tanh)
        self.ac.to(self.args.device)
        self.pi_net_opt = Adam(self.ac.pi_net.parameters(), lr=self.args.pi_lr)
        self.v_net_opt = Adam(self.ac.v_net.parameters(), lr=self.args.vf_lr)
        self.buffer = OnPolicyBuffer()
        self.logger = EpochLogger()

    def train(self):
        eval_reward_list = []
        start_time = time.time()
        total_step = 0
        for epoch in range(self.args.epochs):
            step = 0
            ep_ret, ep_len = 0, 0
            obs = self.env.reset()
            while True:
                step += 1
                action, v_value, log_prob = self.ac.step(torch.as_tensor(obs, dtype=torch.float32))
                self.logger.store(VVals=v_value)
                next_obs, reward, done, _ = self.env.step(action)  # gym似乎会自动截断超出范围的动作
                self.buffer.add(obs, v_value.item(), action, log_prob, reward, done)
                ep_ret += reward
                ep_len += 1
                obs = next_obs
                if done:
                    self.logger.store(EpRet=ep_ret, EpLen=ep_len)
                    if step >= self.args.steps_per_epoch:
                        break
                    obs = self.env.reset()
                    ep_ret, ep_len = 0, 0
            total_step += step
            self.update()
            eval_reward = self.evaluate()
            eval_reward_list.append(eval_reward)
            print("epoch", epoch, "eval_reward", eval_reward)

            # log_info
            # self.logger.log_tabular('Epoch', epoch)
            # self.logger.log_tabular('EpRet', with_min_and_max=True)
            # self.logger.log_tabular('EpLen', average_only=True)
            # self.logger.log_tabular('EvalRet', eval_reward)
            # self.logger.log_tabular('VVals', with_min_and_max=True)
            # self.logger.log_tabular('TotalEnvInteracts', total_step)
            # self.logger.log_tabular('LossPi', average_only=True)
            # self.logger.log_tabular('LossV', average_only=True)
            # self.logger.log_tabular('Time', time.time() - start_time)
            # self.logger.dump_tabular()
        # 存储验证奖励用以绘图分析
        np.save(args.eval_reward_save_path, np.array(eval_reward_list))

    def update(self):
        self.buffer.compute_return(self.args.gamma)
        obs, v_value, action, old_log_prob, returns = self.buffer.obs, self.buffer.v_value, self.buffer.action, self.buffer.log_prob, self.buffer.returns
        old_log_prob = torch.as_tensor(np.squeeze(old_log_prob), dtype=torch.float32)
        for i in range(self.args.k_epoch):
            # update pi net
            self.pi_net_opt.zero_grad()
            # log_prob = self.ac.log_prob(torch.as_tensor(obs, dtype=torch.float32), torch.as_tensor(action))
            log_prob = self.ac.log_prob(torch.as_tensor(obs, dtype=torch.float32),
                                        torch.stack(action) if isinstance(action[0], torch.Tensor) else torch.as_tensor(
                                            action))
            advantage_value = torch.as_tensor(np.array(returns) - np.array(v_value), dtype=torch.float32)
            # advantage_value = torch.as_tensor(returns, dtype=torch.float32)
            # gae
            # deltas = returns[:-1] + self.args.gamma * v_value[1:] - v_value[:-1]
            # adv_buf = core.discount_cumsum(deltas, self.args.gamma * 0.97)
            cur_rate = torch.exp(log_prob - old_log_prob)
            # pi_loss = -(log_prob * advantage_value).mean()
            pi_loss = -torch.min(cur_rate * advantage_value,
                                 torch.clip(cur_rate, 1 - self.args.epsilon, 1 + self.args.epsilon) * advantage_value).mean()
            pi_loss.backward()
            self.pi_net_opt.step()
        # update v net
        for i in range(5):
            self.v_net_opt.zero_grad()
            cur_v_value = self.ac.v_net(torch.as_tensor(obs, dtype=torch.float32))
            vf_loss = (torch.as_tensor(returns, dtype=torch.float32) - torch.squeeze(cur_v_value, 1)).pow(2)
            vf_loss = vf_loss.mean()
            vf_loss.backward()
            self.v_net_opt.step()
        self.logger.store(LossPi=pi_loss.item(), LossV=vf_loss.item())
        self.buffer.clear_buffer()

    def evaluate(self):
        total_reward = []
        for epoch in range(10):
            epoch_reward = []
            obs = self.env.reset()
            while True:
                # action, _, _ = self.ac.pi_net.sample(torch.from_numpy(np.expand_dims(obs.astype(np.float32), 0)))
                action = self.ac.act(torch.as_tensor(obs, dtype=torch.float32))
                obs, reward, done, _ = self.env.step(action)
                epoch_reward.append(reward if isinstance(reward, float) else reward.item())
                if done:
                    break
            total_reward.append(sum(epoch_reward))
        eval_reward = sum(total_reward) / len(total_reward)
        return eval_reward

# todo list
# 内存泄漏问题，为何内存占用在一直增长：解决了绝大部分内存泄露问题，模型运行后，不需要计算梯度的时候，最好用torch.no_grad模式启动，并且尽量不要输出tensor变量到其他地方进行计算
# k_epoch早停机制，在当前策略和采样策略相差较大时，停止使用数据
# 标准的ppo还会把entropy加入loss当中
# 使用GAE优势函数进行测试

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.epochs = 500
    args.steps_per_epoch = 5000
    args.gamma = 0.99
    args.hidden_sizes = [64, 64]
    args.epsilon = 0.2
    args.k_epoch = 5
    args.pi_lr = 3e-4
    args.vf_lr = 1e-3
    args.env_name = 'HalfCheetah-v2'
    args.eval_reward_save_path = './test_result/ppo/'+args.env_name+str(time.localtime()[1:6])
    args.device = 'cpu'

    env = gym.make(args.env_name)
    ppo = PPO(args, env)
    ppo.train()
    ppo.evaluate()
