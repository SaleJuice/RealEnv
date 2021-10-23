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
from cartpole import CartPoleEnv


def test(obs, rew, d, info=None):
    if obs[0] > 4:
        act = 0
    elif obs[0] < -4:
        act = 1

    return act

def middle(obs, rew, d, info=None):



if __name__ == '__main__':
    print("Program Start!")
    env = CartPoleEnv("COM4")
    env.reset()
    action = random.randint(0, 1)
    while True:
        # observation, reward, done, _ = env.step(action, ctrl_freq=100)
        observation, reward, done, _ = env.observe()
        print(observation)
        # action = test(observation, reward, done)
    env.close()
    print("Program Exit!")


# if __name__ == '__main__':
#     print("Program Start!")
#     env = CartPoleEnv("COM4")
#     try:
#         env.reset()
#         action = random.randint(0, 1)
#         while True:
#             # observation, reward, done, _ = env.step(action, ctrl_freq=100)
#             observation, reward, done, _ = env.observe()
#             print(observation)
#             # action = test(observation, reward, done)
#     except:
#         env.close()
#         print("Program Exit!")
