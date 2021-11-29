"""
@Author: Lrw
@Time: 2021/9/30
功能说明: 实现ppo clip版本的算法
"""
import os
import sys
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)
# split only split last value
sys.path.append(os.path.split(base_dir)[0])

import argparse
from ray_imp.wandb_wrapper import *
from ray_imp.common import *
from torch.optim import Adam


class PPO(rl_algorithm):
    def __init__(self, args, env: gym.Env):
        self.env = env
        obs_dim = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space
        self.args = args
        self.ac = MlpActorCritic(obs_dim, self.args.hidden_sizes, self.action_space, nn.Tanh)
        print(self.ac)
        self.ac.to(self.args.device)
        self.pi_net_opt = Adam(self.ac.pi_net.parameters(), lr=self.args.pi_lr)
        self.v_net_opt = Adam(self.ac.v_net.parameters(), lr=self.args.vf_lr)
        if self.args.load_model_flag:
            self.ac.load_state_dict(torch.load(self.args.load_model_path))
        self.buffer = OnPolicyBuffer()

    def train(self):
        eval_reward_list = []
        total_step = 0
        total_epoch = 0
        for epoch in range(self.args.epochs):
            step = 0
            ep_ret, ep_len = 0, 0
            obs = self.env.reset()
            while True:
                step += 1
                action, v_value, log_prob = self.ac.step(torch.as_tensor(obs, dtype=torch.float32))
                next_obs, reward, done, _ = self.env.step(action)  # gym似乎会自动截断超出范围的动作
                self.buffer.add(obs, v_value.item(), action, log_prob, reward, done)
                ep_ret += reward
                ep_len += 1
                obs = next_obs
                if done:
                    total_epoch += 1
                    if step >= self.args.steps_per_epoch:
                        break
                    obs = self.env.reset()
                    ep_ret, ep_len = 0, 0
            total_step += step
            # pi_loss, vf_loss = self.update()
            pi_loss, vf_loss = self.update_together()
            eval_reward = self.evaluate()
            eval_reward_list.append(eval_reward)
            print("epoch", epoch, "eval_reward", eval_reward, "real_epoch", total_epoch)
            wandb.log({"batch_epoch": epoch, "eval_reward": eval_reward, "real_epoch": total_epoch, "pi_loss": pi_loss,
                       'vf_loss': vf_loss}, step=total_step)

            # 保存模型
            if epoch % self.args.save_model_interval == 0:
                torch.save(self.ac.state_dict(), wandb.run.dir + '/models/ac_{}.pth'.format(epoch))
            # torch.onnx.export(self.ac.pi_net, torch.randn(10, self.env.observation_space.shape[0], device='cpu'), 'ppo_model.onnx')

    def update(self):
        # wandb.watch(self.ac, log_freq=5)
        self.buffer.compute_return(self.args.gamma)
        obs, v_value, action, old_log_prob, returns = self.buffer.obs, self.buffer.v_value, self.buffer.action, self.buffer.log_prob, self.buffer.returns
        old_log_prob = torch.as_tensor(np.squeeze(old_log_prob), dtype=torch.float32)
        pi_loss_list, vf_loss_list = [], []
        for i in range(self.args.k_epoch):
            # update pi net
            self.pi_net_opt.zero_grad()
            log_prob = self.ac.log_prob(torch.as_tensor(obs, dtype=torch.float32),
                                        torch.stack(action) if isinstance(action[0],
                                                                          torch.Tensor) else torch.as_tensor(
                                            action))
            if self.args.use_gae:
                batch_size = len(self.buffer.reward)
                advantage_value = torch.zeros(batch_size)
                last_gae = 0
                for t in reversed(range(batch_size)):
                    if t == batch_size - 1:
                        next_done = 1 - self.buffer.done[t]
                        assert self.buffer.done[t], "last done flag is {}!".format(self.buffer.done[t])
                        next_value = 0  # 这里结束时并没有保存下一次的值，暂取0， 因为结束时，都是done
                    else:
                        next_done = 1 - self.buffer.done[t]
                        next_value = v_value[t + 1]
                    delta = self.buffer.reward[t] + next_done * self.args.gamma * next_value - v_value[t]

                    advantage_value[t] = last_gae = delta + self.args.gamma * self.args.gae_lambda * last_gae
                    # use advantage to remake returns
                    # returns = advantage_value + tensor_wrap(v_value)
            else:
                advantage_value = torch.as_tensor(np.array(returns) - np.array(v_value), dtype=torch.float32)
                # 直接用回报，不减去基线
                # advantage_value = torch.as_tensor(returns, dtype=torch.float32)
            # normalization advantage
            # print('normalization advantage')
            advantage_value = (advantage_value - advantage_value.mean()) / (advantage_value.std() + 1e8)
            # 早停机制
            approx_kl = (old_log_prob - log_prob).mean().item()
            if approx_kl > self.args.max_approx_kl:
                print('early stop k_epoch = {}'.format(i))
                break
            cur_rate = torch.exp(log_prob - old_log_prob)
            # pi_loss = -(log_prob * advantage_value).mean()
            pi_loss = -torch.min(cur_rate * advantage_value,
                                 torch.clamp(cur_rate, 1 - self.args.epsilon,
                                             1 + self.args.epsilon) * advantage_value).mean()
            pi_loss.backward()
            nn.utils.clip_grad_norm_(self.ac.pi_net.parameters(), self.args.max_grad_norm)
            self.pi_net_opt.step()
            pi_loss_list.append(pi_loss.item())
        # update v net
        for i in range(5):
            self.v_net_opt.zero_grad()
            cur_v_value = self.ac.v_net(torch.as_tensor(obs, dtype=torch.float32))
            vf_loss = (torch.as_tensor(returns, dtype=torch.float32) - torch.squeeze(cur_v_value, 1)).pow(2)
            vf_loss = vf_loss.mean()
            vf_loss.backward()
            nn.utils.clip_grad_norm_(self.ac.v_net.parameters(), self.args.max_grad_norm)
            self.v_net_opt.step()
            vf_loss_list.append(vf_loss.item())
        self.buffer.clear_buffer()
        return sum(pi_loss_list) / len(pi_loss_list), sum(vf_loss_list) / len(vf_loss_list)

    def update_together(self):
        # wandb.watch(self.ac, log_freq=5)
        self.buffer.compute_return(self.args.gamma)
        obs, v_value, action, old_log_prob, returns = self.buffer.obs, self.buffer.v_value, self.buffer.action, self.buffer.log_prob, self.buffer.returns
        old_log_prob = torch.as_tensor(np.squeeze(old_log_prob), dtype=torch.float32)
        pi_loss_list, vf_loss_list = [], []
        for i in range(self.args.k_epoch):
            # update pi net
            log_prob = self.ac.log_prob(torch.as_tensor(obs, dtype=torch.float32),
                                        torch.stack(action) if isinstance(action[0],
                                                                          torch.Tensor) else torch.as_tensor(
                                            action))
            if self.args.use_gae:
                batch_size = len(self.buffer.reward)
                advantage_value = torch.zeros(batch_size)
                last_gae = 0
                for t in reversed(range(batch_size)):
                    if t == batch_size - 1:
                        next_done = 1 - self.buffer.done[t]
                        assert self.buffer.done[t], "last done flag is {}!".format(self.buffer.done[t])
                        next_value = 0  # 这里结束时并没有保存下一次的值，暂取0， 因为结束时，都是done
                    else:
                        next_done = 1 - self.buffer.done[t]
                        next_value = v_value[t + 1]
                    delta = self.buffer.reward[t] + next_done * self.args.gamma * next_value - v_value[t]

                    advantage_value[t] = last_gae = delta + self.args.gamma * self.args.gae_lambda * last_gae
                    # use advantage to remake returns
                    # returns = advantage_value + tensor_wrap(v_value)
            else:
                advantage_value = torch.as_tensor(np.array(returns) - np.array(v_value), dtype=torch.float32)
                # 直接用回报，不减去基线
                # advantage_value = torch.as_tensor(returns, dtype=torch.float32)
            # normalization advantage
            # print('normalization advantage')
            advantage_value = (advantage_value - advantage_value.mean()) / (advantage_value.std() + 1e8)
            # 早停机制
            approx_kl = (old_log_prob - log_prob).mean().item()
            if approx_kl > self.args.max_approx_kl:
                print('early stop k_epoch = {}'.format(i))
                break
            cur_rate = torch.exp(log_prob - old_log_prob)
            # pi_loss = -(log_prob * advantage_value).mean()
            pi_loss = -torch.min(cur_rate * advantage_value,
                                 torch.clamp(cur_rate, 1 - self.args.epsilon,
                                             1 + self.args.epsilon) * advantage_value).mean()

            cur_v_value = self.ac.v_net(torch.as_tensor(obs, dtype=torch.float32))
            vf_loss = (torch.as_tensor(returns, dtype=torch.float32) - torch.squeeze(cur_v_value, 1)).pow(2).mean()
            loss = pi_loss + 0.5 * vf_loss
            self.pi_net_opt.zero_grad()
            self.v_net_opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.ac.pi_net.parameters(), self.args.max_grad_norm)
            nn.utils.clip_grad_norm_(self.ac.v_net.parameters(), self.args.max_grad_norm)
            self.pi_net_opt.step()
            self.v_net_opt.step()
            pi_loss_list.append(pi_loss.item())
            vf_loss_list.append(0.5 * vf_loss.item())
        self.buffer.clear_buffer()
        return sum(pi_loss_list) / len(pi_loss_list), sum(vf_loss_list) / len(vf_loss_list)

    def evaluate(self):
        total_reward = []
        for epoch in range(10):
            epoch_reward = []
            obs = self.env.reset()
            while True:
                # self.env.render()
                action = self.ac.act(torch.as_tensor(obs, dtype=torch.float32))
                obs, reward, done, _ = self.env.step(wrap_action(action))
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
    args.use_gae = True
    args.gae_lambda = 0.95
    args.hidden_sizes = [64, 64]
    args.epsilon = 0.2
    args.k_epoch = 80  # 5
    args.max_approx_kl = 0.01 * 1.5  # 自动控制训练数据使用轮数，使用近似的kl散度进行计算
    args.pi_lr = 3e-4
    args.vf_lr = 1e-3
    args.max_grad_norm = 0.5
    args.env_name = 'HalfCheetah-v2'
    # args.env_name = 'BipedalWalker-v3'
    # args.env_name = "CartPole-v0"
    args.test_name = 'ppo_pi_vf_together'
    args.test_info = 'use gae and max grad norm and advantage, pi and vf trained together'
    args.save_model_interval = 10
    args.load_model_flag = False
    if args.load_model_flag:
        args.load_model_path = None
    args.device = 'cpu'

    wandb.init(project="HalfCheetah-benchmark")
    config_wandb(wandb, args)

    env = gym.make(args.env_name)
    ppo = PPO(args, env)
    ppo.train()
    ppo.evaluate()
