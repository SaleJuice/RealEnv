# _*_ coding: utf-8 _*_
# @File        : real_cartpole.py
# @Time        : 2021/10/17 19:09
# @Author      : SaleJuice
# @E-Mail      : linxzh@shanghaitech.edu.cn
# @Institution : LIMA Lab, ShanghaiTech University, China
# @SoftWare    : PyCharm


import time
import math
import pprint
import random
import numpy as np
import gym
from gym import spaces, logger
from gym.utils import seeding
from easyserial import WindowsBackground


def rangelimit(cur, lower, upper):
    if cur >= upper:
        cur = upper
    elif cur <= lower:
        cur = lower
    return cur


class RealCartPoleEnv(gym.Env):

    def __init__(self, portx):
        self.ser = WindowsBackground()
        assert (self.ser.connect(portx)), f"Can't connect real env by '{portx}'!"

        self.theta_threshold_radians = 24 * (math.pi / 180)
        self.x_threshold = 5
        high = np.array(
            [
                self.x_threshold,
                np.finfo(np.float32).max,
                self.theta_threshold_radians,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(-high, high, dtype=np.float16)

        self.seed()
        self.state = None
        self.steps_beyond_done = None
        # system param
        self.__x = 0
        self.__px = 0
        self.__dx = 0
        self.__a = 0
        self.__pa = 0
        self.__da = 0
        self.__k = 0
        self.__p = 999
        # time param
        self.start_time = 0
        self.end_time = 0
        # offset
        self.__ox = 0
        self.__oa = 3648

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def __input(self):
        num, res, _ = self.ser.read("all")
        if num >= 16 and res[res.rfind(')')-14] == '(':
            res = res[res.rfind(')')-13:res.rfind(')')]
            raw_data = list(map(int, res.split(',')))

            self.__x = (raw_data[0] - self.__ox) * 10 / 134200
            self.__a = (raw_data[1] - self.__oa) * math.pi / 4096
            self.__k = raw_data[2]
            self.__dx = self.__x - self.__px
            self.__da = self.__a - self.__pa
            self.__px = self.__x
            self.__pa = self.__a
            self.state = [self.__x, self.__dx, self.__a, self.__da]
            return True
        return False

    def __output(self):
        if self.ser.write(f"({self.__p},0,0)\n"):
            return True
        return False

    def step(self, action, ctrl_freq=20):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        # if action:
        #     self.__p = 1000
        # else:
        #     self.__p = -1000

        # if action:
        #     self.__p += 300
        # else:
        #     self.__p += -300

        print(action)

        if action == 0:
            self.__p += -400
        elif action == 1:
            self.__p += -200
        elif action == 2:
            self.__p += 0
        elif action == 3:
            self.__p += 200
        elif action == 4:
            self.__p += 400

        if self.__p >= 1800:
            self.__p = 1800
        elif self.__p <= -1800:
            self.__p = -1800

        self.__output()
        while self.end_time - self.start_time < ctrl_freq/1000:
            self.end_time = time.perf_counter()
        self.start_time = time.perf_counter()
        self.__input()

        done = bool(
            self.state[0] < -self.x_threshold+1
            or self.state[0] > self.x_threshold-1
            or self.state[2] < -self.theta_threshold_radians
            or self.state[2] > self.theta_threshold_radians
        )

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state, dtype=np.float16), reward, done, {}

    def reset(self, touch=False):
        if touch:
            # touch edge
            self.__input()
            while self.__k != 1:
                self.__input()
                self.__p = -800
                self.__output()
            self.__ox = (self.__x * 134200 / 10) + 67100  # 66569
            self.__p = 800
            self.__output()
        # back middle
        self.__input()
        randomevent = 0.05 * random.randint(-10, 10)
        while abs(self.__x - randomevent) >= 0.01:
            self.__input()
            if abs(self.__x - randomevent) >= 1.0:
                pwm = 800
            else:
                pwm = 400
            if self.__x < randomevent:
                self.__p = pwm
            else:
                self.__p = -pwm
            self.__output()
        for _ in range(3):
            self.__p = 0
            self.__output()
        # return
        print("Press the bottom to reset the real env!")
        while self.__k != 2:
            self.__input()
        self.__input()
        self.steps_beyond_done = None
        return np.array(self.state, dtype=np.float16)

    def render(self, mode='human'):
        pass

    def close(self):
        self.__p = 0
        self.__output()


if __name__ == '__main__':
    print("Program Start!")
    env = CartPoleEnv("COM4")
    try:
        env.reset()
        action = random.randint(0, 1)
        while True:
            observation, reward, done, _ = env.step(action, ctrl_freq=100)
            print(observation)
            if observation[0] > 4:
                action = 0
            elif observation[0] < -4:
                action = 1
    except:
        env.close()
        print("Program Exit!")



