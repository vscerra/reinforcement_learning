# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 13:54:20 2025

@author: vscerra
"""
import numpy as np

class Greedy:
    '''Greedy algorithm - only selects the action with the largest cumulative reward value at any time t'''
    def __init__(self, k = 10):
        self.k = k
        self.Q = np.zeros(k) # estimated values
        self.N = np.zeros(k) # action counts
        
    def select_action(self):
        ''' select action with greedy strategy'''
        return np.argmax(self.Q) # always exploit
      
    def update(self, action, reward):
        ''' update Q value estimates based on observed reward '''
        self.N[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.N[action]
        

class EpsilonGreedy:
    ''' Epsilon-Greedy  - selects the greedy option 1-epsilon proportion of the time, 
    and for epsilon proportion of the trials, it selects randomly from all possible actions
    adjust epsilon parameter and optional optimistic Q0 '''
    def __init__(self, k = 10, epsilon = 0.1, Q0 = 0):
        self.k = k
        self.epsilon = epsilon
        self.Q = np.ones(k) * Q0 # initial optimistic values
        self.N = np.zeros(k) # action counts
        
    def select_action(self):
        ''' select an action using an epsilon-greedy strategy '''
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.k) # explore
        return np.argmax(self.Q) # exploit
      
    def update(self, action, reward):
        ''' update Q-value estimates based on observed reward '''
        self.N[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.N[action]
        

class GradientBandit:
    ''' Gradient bandit algorithm - optimizes the reward function via stochatic gradient ascent
    using a softmax policy, adjusting action preferences to favor actions with higher rewards
    adjust alpha parameter'''
    def __init__(self, k = 10, alpha = 0.1):
        self.k = k
        self.alpha = alpha
        self.H = np.zeros(k) # Preferences
        self.pi = np.ones(k) / k # initial uniform probabilities
        self.avg_reward = 0 # baseline reward
        self.t = 0 # time step

    def select_action(self):
        ''' selects an action based on softmax preferences '''
        exp_H = np.exp(self.H - np.max(self.H)) # for numerical stabiilty
        self.pi = exp_H / np.sum(exp_H)
        return np.random.choice(self.k, p = self.pi)
      
    def update(self, action, reward):
        ''' updates preferences using gradient ascent '''
        self.t += 1
        self.avg_reward += (reward - self.avg_reward) / self.t # update baseline
        for a in range(self.k):
            if a == action:
                self.H[a] += self.alpha * (reward - self.avg_reward) * (1 - self.pi[a])
            else:
                self.H[a] -= self.alpha * (reward - self.avg_reward) * self.pi[a]
                

class UCB: 
    ''' Upper confidence bound (UCB) algorithm - balances the tradeoff between exploitation and 
    # exploration using confidence intervals. An action will be selected if it has returned good rewards
    # OR if it hasn't been explored much so that the reward estimates are uncertain
    # adjust c '''
    def __init__(self, k = 10, c = 2):
        self.k = k
        self.c = c
        self.Q = np.zeros(k) # Estimated values
        self.N = np.zeros(k) # action counts
        self.t = 0 # time step
      
    def select_action(self):
        ''' selects an action using UCB formula '''
        self.t += 1
        if  self.t <= self.k: # ensure each action is tried at least once
            return self.t - 1
          
        ucb_values = self.Q + self.c * np.sqrt(np.log(self.t) / (self.N + 1e-5)) # prevent dividing by zero
        return np.argmax(ucb_values)
      
    def update(self, action, reward):
        ''' Updates Q-value estimates based on observed rewards '''
        self.N[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.N[action]
        

class ThompsonSampling:
    """ Thompson Sampling for multi-armed bandits (Binary Rewards - Beta distribution) """
    def __init__(self, k = 10, alpha = 1, beta = 1):
        self.k = k
        self.alpha = np.ones(k) * alpha # success count
        self.beta = np.ones(k) * beta # failures count
        
    def select_action(self):
        """select an action by sampling from the beta distribution"""
        samples = np.random.beta(self.alpha, self.beta)
        return np.argmax(samples)
      
    def update(self, action, reward):
        """ update beta distribution parameters based on recieved reward"""
        self.alpha[action] = max(1, self.alpha[action] + reward)  # Keep α >= 1
        self.beta[action] = max(1, self.beta[action] + (1 - reward))  # Keep β >= 1
        
        