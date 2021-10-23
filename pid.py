# _*_ coding: utf-8 _*_
# @File        : pid.py
# @Time        : 2021/10/23 20:03
# @Author      : SaleJuice
# @E-Mail      : linxzh@shanghaitech.edu.cn
# @Institution : LIMA Lab, ShanghaiTech University, China
# @SoftWare    : PyCharm

import time
import math
import random
import pprint
import numpy as np


def rangelimit(cur, lower, upper):
    if cur >= upper:
        cur = upper
    elif cur <= upper:
        cur = lower
    return cur


class PositionPid:

    def __init__(self, kp, ki, kd, dt=20, ):
        self.dt = dt  # ms

        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.limit_err_sum = 0
        self.limit_out_i = 0
        self.limit_out_all = 0

        self.err_cur = 0
        self.err_last = 0
        self.err_sum = 0

        self.ep = 0
        self.ei = 0
        self.ed = 0

        self.out_p = 0
        self.out_i = 0
        self.out_d = 0
        self.out_all = 0

    def calculate(self, exp, cur):
        self.err_cur = exp - cur

        self.err_sum = rangelimit(self.err_sum + self.err_cur, -self.limit_err_sum, self.limit_err_sum)

        self.ep = self.err_cur
        self.ei = self.err_sum
        self.ed = self.err_cur - self.err_last

        self.out_p = self.kp * self.ep
        self.out_i = self.ki * self.ei
        self.out_d = self.kd * self.ed

        self.out_all = self.out_p + self.out_i + self.out_d
        self.err_last = self.err_cur

        return self.out_all

    def clean(self):
        self.err_cur = 0
        self.err_last = 0
        self.err_sum = 0

        self.ep = 0
        self.ei = 0
        self.ed = 0

        self.out_p = 0
        self.out_i = 0
        self.out_d = 0
        self.out_all = 0

