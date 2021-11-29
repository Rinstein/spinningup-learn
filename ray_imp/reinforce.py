import argparse
import gym
import torch
from torch import nn
from torch.optim import Adam
from torch.distributions import Categorical
import numpy as np
from ray_imp.common import *


class MlpCategoricalActor(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, action_dim, hidden_act, out_act=nn.Identity):
        super().__init__()
        self.net = nn.Sequential(
            *get_mlp_layers(obs_dim, hidden_sizes, action_dim, hidden_act, out_act)
        )

    def forward(self, x):
        return self.net(x)

    def log_prob(self, obs, action):
        logits_out = self(obs)
        act_dis = Categorical(logits=logits_out)
        return act_dis.log_prob(action)

    def act(self, x):
        out = self(x)
        act_dis = Categorical(logits=out)
        action = act_dis.sample()
        return action.item()


class Reinforce(object):
    def __init__(self, args, env: gym.Env):
        self.env = env
        assert isinstance(self.env.action_space, gym.spaces.Discrete)
        assert isinstance(self.env.observation_space, gym.spaces.Box)
        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.n
        self.args = args
        self.actor = MlpCategoricalActor(obs_dim, self.args.hidden_sizes, act_dim, nn.Tanh)
        self.actor.to(self.args.device)
        self.actor_opt = Adam(self.actor.parameters(), lr=self.args.lr)

    def train(self):
        total_epoch = 0
        for epoch in range(self.args.epochs):
            batch_obs, batch_action, batch_return = [], [], []
            step = 0
            ep_ret, ep_len = 0, 0
            obs = self.env.reset()
            real_epoch = 0
            real_epoch_reward = 0
            while True:
                step += 1
                action = self.actor.act(torch.as_tensor(obs, dtype=torch.float32))
                next_obs, reward, done, _ = self.env.step(action)
                batch_obs.append(obs)
                batch_action.append(action)
                ep_ret += reward
                ep_len += 1
                obs = next_obs
                if done:
                    real_epoch += 1
                    real_epoch_reward += ep_ret
                    batch_return += [ep_ret]*ep_len
                    if step >= self.args.steps_per_epoch:
                        break
                    obs = self.env.reset()
                    ep_ret, ep_len = 0, 0
            self.actor.zero_grad()
            log_prob = self.actor.log_prob(torch.as_tensor(batch_obs, dtype=torch.float32), torch.as_tensor(batch_action))
            pi_loss = -(log_prob * torch.as_tensor(batch_return, dtype=torch.float32)).mean()
            pi_loss.backward()
            self.actor_opt.step()
            print("epoch", epoch, "batch_returns", np.mean(batch_return))
            total_epoch += real_epoch
            print("total_epoch", total_epoch, "real_epoch_reward", real_epoch_reward/real_epoch)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.epochs = 100
    args.steps_per_epoch = 5000
    args.gamma = 1
    args.hidden_sizes = [32]
    args.lr = 1e-2
    args.device = 'cpu'

    # env = gym.make('HalfCheetah-v2')
    env = gym.make('CartPole-v0')
    reinforce = Reinforce(args, env)
    reinforce.train()
