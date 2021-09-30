import gym
import torch
from torch import nn
from torch.distributions import Categorical, Normal
from abc import ABCMeta, abstractmethod


def wrap_action(action: torch.Tensor):
    if action.numel() == 1:
        return action.item()
    return action


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
        return wrap_action(action), act_dis.log_prob(action).detach().numpy()

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
        return wrap_action(action), act_dis.log_prob(action).sum(axis=-1).detach().numpy()

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
        action, log_prob = self.pi_net.act(x)
        v_value = self.v_net(x)
        return action, v_value, log_prob

    def log_prob(self, obs, action):
        return self.pi_net.log_prob(obs, action)


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
