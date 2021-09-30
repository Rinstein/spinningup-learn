from torch import nn


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