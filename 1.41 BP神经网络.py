#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2019/3/21 19:12
#@Author: KartalLee
import numpy as np
sizes=[2,3,2]
for y in sizes[1:]:
    print(y)
for x, y in zip(sizes[:-1], sizes[1:]):
    print(y,x)
biases = [np.random.randn(y, 1) for y in sizes[1:]]
weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
print(biases)
print(weights)