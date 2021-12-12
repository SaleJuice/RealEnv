# _*_ coding: utf-8 _*_
# @File        : demo.py
# @Time        : 2021/10/23 19:16
# @Author      : SaleJuice
# @E-Mail      : linxzh@shanghaitech.edu.cn
# @Institution : LIMA Lab, ShanghaiTech University, China
# @SoftWare    : PyCharm

import time
import math
import random
import pprint
import numpy as np
import gym
from gym import spaces, logger
from gym.utils import seeding
from sim_cartpole import SimCartPoleEnv
from real_cartpole_swingup import RealCartPoleSwingUpEnv
import torch
from tianshou.utils.net.common import Net
from tianshou.data import Collector, ReplayBuffer, PrioritizedReplayBuffer
from tianshou.policy import DQNPolicy
from tianshou.trainer import offpolicy_trainer
from trainer import my_trainer



def debug():
    print("Program Start!")
    env = CartPoleEnv("COM4")
    env.reset()
    env.close()
    print("Program Exit!")


def demo():
    print("Program Start!")
    env = CartPoleEnv("COM7")
    try:
        env.reset()
        action = random.randint(0, 1)
        while True:
            observation, reward, done, _ = env.step(action, ctrl_freq=20)
            if observation[0] > 4:
                action = 0
            elif observation[0] < -4:
                action = 1
            print(observation, action, reward, done)
    except:  # KeyboardInterrupt:
        env.close()
        print("Program Exit!")


def drl():
    print("Program Start!")
    env = CartPoleEnv("COM4")
    try:
        env.reset()
        while True:
            observation, reward, done, _ = env.step(action, ctrl_freq=20)
    except:  # KeyboardInterrupt:
        env.close()
        print("Program Exit!")


def sim():
    # env √
    print("env")
    # env = SimCartPoleEnv()
    env = RealCartPoleSwingUpEnv("COM5")
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n

    # model √
    print("model")
    net = Net(state_shape, action_shape, hidden_sizes=[128, 128, 128], device='cpu').to('cpu')
    # optim = torch.optim.Adam(net.parameters(), lr=1e-1)
    optim = torch.optim.RMSprop(net.parameters(), lr=1e-2)

    # policy √
    print("policy")
    policy = DQNPolicy(model=net, optim=optim, discount_factor=0.999, estimation_step=3, target_update_freq=10, is_double=True)
    # policy.load_state_dict(torch.load("policy_11_9_40epoch.pth"))

    # buffer √
    print("buffer")
    buf = PrioritizedReplayBuffer(size=10000, alpha=0.6, beta=0.4)

    # collector √
    print("collector")
    train_collector = Collector(policy=policy, env=env, buffer=buf, exploration_noise=True)
    test_collector = Collector(policy=policy, env=env, exploration_noise=True)

    # # prepare √
    # print("prepare")
    # result = test_collector.collect(n_step=2000, render=True, random=True)
    # pprint.pprint(result)
    # exit()

    # train_fn
    def train_fn(epoch, env_step):
        if env_step <= 500:
            policy.set_eps(0.1)
        elif env_step <= 2000:
            eps = 0.1 - (env_step - 500) / 1500 * (0.9 * 0.1)
            policy.set_eps(eps)
        else:
            policy.set_eps(0.1 * 0.1)

    def test_fn(epoch, env_step):
        policy.set_eps(0)

    def stop_fn(mean_rewards):
        return mean_rewards >= 1000

    def save_fn(policy):
        torch.save(policy.state_dict(), 'policy.pth')
    print("train_fn")

    # trainer
    print("training")
    result = offpolicy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=None,
        test_in_train=False,
        max_epoch=500,
        step_per_epoch=500,
        step_per_collect=10,
        episode_per_test=1,
        batch_size=256,
        update_per_step=3,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_fn=save_fn
    )


if __name__ == '__main__':
    print("Start!")
    # debug()
    # demo()
    # drl()
    sim()
    print("Finish!")
