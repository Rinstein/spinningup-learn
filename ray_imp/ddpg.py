"""
@Author: Lrw
@Time: 2021/10/9
功能说明: ddpg算法的实现
"""
import time
import argparse
import copy
from ray_imp.wandb_wrapper import *
from ray_imp.common import *
from torch.optim import Adam


class DDPG(rl_algorithm):
    """DDPG仅仅适用于连续动作空间"""

    def __init__(self, args, env: gym.Env):
        self.env = env
        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.action_space = self.env.action_space
        self.min_action = self.action_space.low[0]
        self.max_action = self.action_space.high[0]
        self.args = args
        self.ac = DDPGActorCritic(self.obs_dim, self.args.hidden_sizes, self.action_space.shape[0], nn.ReLU)
        print(self.ac)
        self.ac_target = copy.deepcopy(self.ac)
        self.ac.to(self.args.device)
        self.pi_net_opt = Adam(self.ac.pi_net.parameters(), lr=self.args.pi_lr)
        self.q_net_opt = Adam(self.ac.q_net.parameters(), lr=self.args.qf_lr)
        self.buffer = OffPolicyBuffer(self.obs_dim, self.action_dim, args.max_buffer_size)

    def train(self):
        total_epoch = 0
        total_step = 0
        for epoch in range(self.args.epochs):
            step = 0
            ep_ret, ep_len = 0, 0
            obs = self.env.reset()
            while True:
                step += 1
                total_step += 1
                if total_step < self.args.start_step:
                    action = self.env.action_space.sample()
                else:
                    action = self.ac.step(tensor_wrap(obs))
                    action += self.args.act_noise * np.random.randn(self.action_dim)
                    action = np.clip(action, self.min_action, self.max_action)
                next_obs, reward, done, _ = self.env.step(action)
                self.buffer.add(obs, action, reward, next_obs, done)
                if (self.args.update_after < total_step) and (total_step % self.args.update_every == 0):
                    pi_loss, q_loss = self.update()
                    wandb.log({'q_loss': q_loss, 'pi_loss': pi_loss}, step=total_step)
                ep_ret += reward
                ep_len += 1
                obs = next_obs
                if done:
                    total_epoch += 1
                    if step >= self.args.steps_per_epoch:
                        break
                    obs = self.env.reset()
                    ep_ret, ep_len = 0, 0
            eval_reward = self.evaluate()
            print("epoch", epoch, "eval_reward", eval_reward, 'real_epoch', total_epoch, 'cur buffer size',
                  self.buffer.size)
            wandb.log({"epoch": epoch, "eval_reward": eval_reward, 'real_epoch': total_epoch, 'cur buffer size':
                self.buffer.size}, step=total_step)

            if epoch % args.save_model_interval == 0:
                torch.save(self.ac.state_dict(), wandb.run.dir + '/models/ac_{}.pth'.format(epoch))

    def update(self):
        loss_q_list, loss_pi_list = [], []
        for up_index in range(self.args.update_every):
            obs, action, reward, next_obs, done = self.buffer.sample(self.args.batch_size)
            # 更新Q网络
            target_q = self.ac_target.get_target_q(tensor_wrap(next_obs))  # 该写法和下面一种写法是一样的
            # with torch.no_grad():
            #     target_action = self.ac_target.act(tensor_wrap(next_obs))
            #     target_q = self.ac_target.get_cur_q(tensor_wrap(next_obs), target_action)
            self.q_net_opt.zero_grad()
            cur_q = self.ac.get_cur_q(tensor_wrap(obs), tensor_wrap(action))
            aim_q = tensor_wrap(reward) + tensor_wrap(self.args.gamma * (1 - done)) * target_q
            loss_q = ((cur_q - aim_q) ** 2).mean()
            loss_q.backward()
            self.q_net_opt.step()
            # 更新Actor网络
            self.pi_net_opt.zero_grad()
            # 实现方式一
            # new_action = self.ac.act(tensor_wrap(obs))
            # loss_pi = -self.ac.get_cur_q(tensor_wrap(obs), new_action).mean()
            # loss_pi.backward()
            # self.pi_net_opt.step()
            # 实现方式二
            for p in self.ac.q_net.parameters():
                p.requires_grad = False
            new_action = self.ac.act(tensor_wrap(obs))
            loss_pi = -self.ac.get_cur_q(tensor_wrap(obs), new_action).mean()
            loss_pi.backward()
            self.pi_net_opt.step()
            for p in self.ac.q_net.parameters():
                p.requires_grad = True
            # 更新target网络
            with torch.no_grad():
                for p, p_targ in zip(self.ac.parameters(), self.ac_target.parameters()):
                    # NB: We use an in-place operations "mul_", "add_" to update target
                    # params, as opposed to "mul" and "add", which would make new tensors.
                    p_targ.data.mul_(self.args.polyak)
                    p_targ.data.add_((1 - self.args.polyak) * p.data)
            loss_q_list.append(loss_q.item())
            loss_pi_list.append(loss_pi.item())
        return sum(loss_pi_list) / len(loss_pi_list), sum(loss_q_list) / len(loss_q_list)

    def evaluate(self):
        total_reward = []
        for epoch in range(10):
            epoch_reward = []
            obs = self.env.reset()
            while True:
                action = self.ac.step(tensor_wrap(obs))
                obs, reward, done, _ = self.env.step(np.clip(action, self.min_action, self.max_action))
                epoch_reward.append(reward if isinstance(reward, float) or isinstance(reward, int) else reward.item())
                if done:
                    break
            total_reward.append(sum(epoch_reward))
        eval_reward = sum(total_reward) / len(total_reward)
        return eval_reward


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.epochs = 500
    args.steps_per_epoch = 5000
    args.gamma = 0.99
    args.hidden_sizes = [256, 256]
    args.pi_lr = 1e-3
    args.qf_lr = 1e-3
    args.act_noise = 0.1
    args.max_buffer_size = int(1e6)
    args.start_step = 10000
    args.batch_size = 100
    args.update_after = 1000
    args.update_every = 50
    args.polyak = 0.995
    # args.env_name = 'HalfCheetah-v2'
    args.env_name = 'BipedalWalker-v3'
    args.test_name = 'ddpg_basic'
    args.save_model_interval = 10
    args.load_model_flag = False
    if args.load_model_flag:
        args.load_model_path = None
    args.device = 'cpu'

    wandb.init(project="BipedalWalker-benchmark")
    wandb.config.update(args)
    config_wandb(wandb, args)

    env = gym.make(args.env_name)
    ddpg = DDPG(args, env)
    ddpg.train()
    ddpg.evaluate()
