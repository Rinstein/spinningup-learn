import gym
import torch
import random
from torch import nn
from torch.distributions import Categorical, Normal
from abc import ABCMeta, abstractmethod
import numpy as np


def wrap_action(action: torch.Tensor):
    if action.numel() == 1:
        return action.item()
    return action.numpy()


def get_mlp_layers(obs_dim, hidden_sizes, out_dim, hidden_act, out_act):
    sizes = [obs_dim] + list(hidden_sizes) + [out_dim]
    layers = []
    for i in range(len(sizes) - 1):
        act_func = hidden_act if i != len(sizes) - 2 else out_act
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        layers.append(act_func())
    return layers


class OnPolicyBuffer(object):
    def __init__(self):
        self.obs = []
        self.v_value = []
        self.action = []
        self.log_prob = []
        self.reward = []
        self.done = []
        self.returns = []

    def add(self, obs, v_value, action, log_prob, reward, done):
        self.obs.append(obs)
        self.v_value.append(v_value)
        self.action.append(action)
        self.log_prob.append(log_prob)
        self.reward.append(reward)
        self.done.append(done)

    def clear_buffer(self):
        del self.obs[:]
        del self.v_value[:]
        del self.action[:]
        del self.log_prob[:]
        del self.reward[:]
        del self.done[:]
        del self.returns[:]

    def compute_return(self, gamma):
        self.returns = [None] * len(self.done)
        cal_returns = 0
        for i in reversed(range(len(self.done))):
            if self.done[i]:
                cal_returns = 0
            cal_returns = self.reward[i] + gamma * cal_returns
            self.returns[i] = cal_returns

    def print_buffer_data_size(self):
        for ele in [self.obs, self.v_value, self.action, self.log_prob, self.reward, self.done, self.returns]:
            if len(ele) > 0:
                print(len(ele))


class OffPolicyBuffer(object):
    def __init__(self, obs_dim, action_dim, max_size):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.obs = np.zeros([max_size, *obs_dim], dtype=np.float32)
        self.action = np.zeros([max_size, *action_dim])
        self.reward = np.zeros([max_size], dtype=np.float32)
        self.next_obs = np.zeros([max_size, *obs_dim], dtype=np.float32)
        self.done = np.zeros([max_size], dtype=np.bool8)

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = obs
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.done[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        assert self.ptr >= batch_size, "采样数据的条数超出了当前回放池的数据"
        sample_index = random.randint(0, self.size, batch_size)
        return self.obs[sample_index], self.action[sample_index], self.reward[sample_index], self.next_obs[
            sample_index], self.done[sample_index]


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
        return action, act_dis.log_prob(action)

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
        return action, act_dis.log_prob(action).sum(axis=-1)

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
        return self.pi_net.act(x)[0]

    def step(self, x):
        with torch.no_grad():
            action, log_prob = self.pi_net.act(x)
            v_value = self.v_net(x)
        return wrap_action(action), v_value.numpy(), log_prob.numpy()

    def log_prob(self, obs, action):
        return self.pi_net.log_prob(obs, action)


class DDPGActorCritic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, action_dim, hidden_act):
        super().__init__()
        self.pi_net = MlpGaussianActor(obs_dim, hidden_sizes, action_dim, hidden_act)
        self.q_net = MlpCritic(obs_dim, hidden_sizes, 1, hidden_act)

    # todo 编写DDPG model
    def step(self, obs):
        with torch.no_grad():
            action = self.pi_net(obs)





class rl_algorithm(metaclass=ABCMeta):

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass
