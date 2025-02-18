# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 13:51:38 2025

@author: vscerra
"""
import numpy as np

class Bandit:
    ''' A multi-armed bandit environment with k-arms '''
    def __init__(self, k = 10):
        self.k = k # number of actions
        self.q_true = np.random.normal(0, 1, k) # True reward values
        
    def get_reward(self, action):
        ''' Returns a noisy reward based on the true value of the selected action '''
        return np.random.normal(self.q_true[action], 1) 
      