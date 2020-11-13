#!/usr/bin/env python

import random
import sys
import copy

import numpy as np

class RandomSeeds():
    def __init__(self,lower, upper, fun):
        self.H = len(lower)
        self.lower = np.array(lower)
        self.upper = np.array(upper)
        self.evaluate = fun
        self.actions = self.lower + (self.upper-self.lower)*np.random.rand(*self.lower.shape) # randomly initialize actions on H steps
        self.value = fun(self.actions)

class RandomShooting():
    def __init__(self, lower, upper, fun, N):
        self.best_value = None
        self.solution = None
        self.lower = lower
        self.upper = upper
        self.fun = fun
        self.N = int(N)

    def run(self):
        candidates = [RandomSeeds(self.lower,self.upper,self.fun) for i in range(self.N)]
        _candidates = sorted(candidates, key=lambda x: x.value)
        self.solution = _candidates[0].actions
        return _candidates[0].value