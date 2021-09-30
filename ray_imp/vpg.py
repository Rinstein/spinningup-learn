import argparse
import time

import gym
import torch
from torch.optim import Adam
from torch.distributions import Categorical, Normal
import numpy as np
from ray_imp.common import *
from spinup.utils.logx import EpochLogger


def wrap_action(action: torch.Tensor):
    if action.numel() == 1:
        return action.item()
    return action

def wrap_tensor_list(obs: list):
    # if isinstance()
    pass


class MlpCategoricalActor(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, action_dim, hidden_act, out_act=nn.Identity):
        super().__init__()
        self.net = nn.Sequential(
            *get_mlp_layers(obs_dim, hidden_sizes, action_dim, hidden_act, out_act)
        )

    def forward(self, x):
        return self.net(x)

    def act(self, obs):
        out = self(obs)
        act_dis = Categorical(logits=out)
        action = act_dis.sample()
        return wrap_action(action)

    def log_prob(self, obs, action):
        out = self(obs)
        act_dis = Categorical(logits=out)
        return act_dis.log_prob(action)


class MlpGaussianActor(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, action_dim, hidden_act, out_act=nn.Identity):
        super().__init__()
        self.log_std = nn.Parameter(-0.5 * torch.ones(action_dim, dtype=torch.float32))
        self.net = nn.Sequential(
            *get_mlp_layers(obs_dim, hidden_sizes, action_dim, hidden_act, out_act)
        )

    def forward(self, x):
        return self.net(x)

    def act(self, obs):
        mu = self(obs)
        std = torch.exp(self.log_std)
        act_dis = Normal(mu, std)
        action = act_dis.sample()
        return wrap_action(action)

    def log_prob(self, obs, action):
        mu = self(obs)
        std = torch.exp(self.log_std)
        act_dis = Normal(mu, std)
        return act_dis.log_prob(action).sum(axis=-1)


class MlpCritic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, action_dim, hidden_act, out_act=nn.Identity):
        super().__init__()
        self.net = nn.Sequential(
            *get_mlp_layers(obs_dim, hidden_sizes, action_dim, hidden_act, out_act)
        )

    def forward(self, x):
        return self.net(x)


class MlpActorCritic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, action_space, hidden_act):
        super().__init__()
        if isinstance(action_space, gym.spaces.Box):
            self.pi_net = MlpGaussianActor(obs_dim, hidden_sizes, action_space.shape[0], hidden_act)
        elif isinstance(action_space, gym.spaces.Discrete):
            self.pi_net = MlpCategoricalActor(obs_dim, hidden_sizes, action_space.n, hidden_act)
        else:
            raise Exception("current action space is not support! Only support Box and Discrete.")
        self.v_net = MlpCritic(obs_dim, hidden_sizes, 1, hidden_act)

    def act(self, x):
        return self.pi_net.act(x)

    def step(self, x):
        action = self.pi_net.act(x)
        v_value = self.v_net(x)
        return action, v_value

    def log_prob(self, obs, action):
        return self.pi_net.log_prob(obs, action)


class VPG(object):
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
        start_time = time.time()
        total_step = 0
        for epoch in range(self.args.epochs):
            step = 0
            ep_ret, ep_len = 0, 0
            obs = self.env.reset()
            while True:
                step += 1
                action, v_value = self.ac.step(torch.as_tensor(obs, dtype=torch.float32))
                self.logger.store(VVals=v_value)
                next_obs, reward, done, _ = self.env.step(action)
                self.buffer.add(obs, v_value.item(), action, None, reward, done)
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
            print("epoch", epoch, "eval_reward", eval_reward)

            #log_info
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

    def update(self):
        self.buffer.compute_return(self.args.gamma)
        obs, v_value, action, returns = self.buffer.obs, self.buffer.v_value, self.buffer.action, self.buffer.returns
        # update pi net
        self.pi_net_opt.zero_grad()
        # log_prob = self.ac.log_prob(torch.as_tensor(obs, dtype=torch.float32), torch.as_tensor(action))
        log_prob = self.ac.log_prob(torch.as_tensor(obs, dtype=torch.float32), torch.stack(action) if isinstance(action[0], torch.Tensor) else torch.as_tensor(action))
        advantage_value = torch.as_tensor(np.array(returns) - np.array(v_value), dtype=torch.float32)
        # advantage_value = torch.as_tensor(returns, dtype=torch.float32)
        # gae
        # deltas = returns[:-1] + self.args.gamma * v_value[1:] - v_value[:-1]
        # adv_buf = core.discount_cumsum(deltas, self.args.gamma * 0.97)
        pi_loss = -(log_prob * advantage_value).mean()
        pi_loss.backward()
        self.pi_net_opt.step()
        # update v net
        for i in range(1):
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.epochs = 1000
    args.steps_per_epoch = 5000
    args.gamma = 0.99
    args.hidden_sizes = [64, 64]
    args.pi_lr = 3e-4
    args.vf_lr = 1e-3
    args.device = 'cpu'

    env = gym.make('HalfCheetah-v2')
    # env = gym.make('CartPole-v0')
    vpg = VPG(args, env)
    vpg.train()
    vpg.evaluate()