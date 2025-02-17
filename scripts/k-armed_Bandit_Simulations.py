# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 13:24:25 2025

@author: vscerra
"""

# Going through the Sutton & Barto (2018) book, this is just a script to simulate the various 
# stationary algorithms presented in Chapter 2

import numpy as np
import matplotlib.pyplot as plt

# multi-armed bandit environment

class Bandit:
    def __init__(self, k = 10):
        self.k = k # number of actions
        self.q_true = np.random.normal(0, 1, k) # True reward values
        
    def get_reward(self, action):
        return np.random.normal(self.q_true[action], 1) # reward with noise
      
  
# Greedy algorithm - only selects the action with the largest cumulative reward value at any time t
class Greedy:
    def __init__(self, k = 10):
        self.k = k
        self.Q = np.zeros(k) # estimated values
        self.N = np.zeros(k) # action counts
        
    def select_action(self):
        return np.argmax(self.Q) # always exploit
      
    def update(self, action, reward):
        self.N[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.N[action]
        
        
# Epsilon-Greedy  - selects the greedy option 1-epsilon proportion of the time, 
# and for epsilon proportion of the trials, it selects randomly from all possible actions
# adjust epsilon parameter and optimistic Q0 

class EpsilonGreedy:
    def __init__(self, k = 10, epsilon = 0.1, Q0 = 0):
        self.k = k
        self.epsilon = epsilon
        self.Q = np.ones(k) * Q0 # initial optimistic values
        self.N = np.zeros(k) # action counts
        
    def select_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.k) # explore
        return np.argmax(self.Q) # exploit
      
    def update(self, action, reward):
        self.N[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.N[action]
        
        
# Gradient bandit algorithm - optimizes the reward function via stochatic gradient ascent
# using a softmax policy, adjusting action preferences to favor actions with higher rewards
# adjust alpha parameter

class GradientBandit:
    def __init__(self, k = 10, alpha = 0.1):
        self.k = k
        self.alpha = alpha
        self.H = np.zeros(k) # Preferences
        self.pi = np.ones(k) / k # initial uniform probabilities
        self.avg_reward = 0 # baseline reward
        self.t = 0 # time step

    def select_action(self):
        exp_H = np.exp(self.H - np.max(self.H)) # for numerical stabiilty
        self.pi = exp_H / np.sum(exp_H)
        return np.random.choice(self.k, p = self.pi)
      
    def update(self, action, reward):
        self.t += 1
        self.avg_reward += (reward - self.avg_reward) / self.t # update baseline
        for a in range(self.k):
            if a == action:
                self.H[a] += self.alpha * (reward - self.avg_reward) * (1 - self.pi[a])
            else:
                self.H[a] -= self.alpha * (reward - self.avg_reward) * self.pi[a]
                
  
# Upper confidence bound (UCB) algorithm - balances the tradeoff between exploitation and 
# exploration using confidence intervals. An action will be selected if it has returned good rewards
# OR if it hasn't been explored much so that the reward estimates are uncertain
# adjust c

class UCB: 
    def __init__(self, k = 10, c = 2):
        self.k = k
        self.c = c
        self.Q = np.zeros(k) # Estimated values
        self.N = np.zeros(k) # action counts
        self.t = 0 # time step
      
    def select_action(self):
        self.t += 1
        if 0 in self.N:
            return np.argmin(self.N) # ensure each action is tried at least once
        ucb_values = self.Q + self.c * np.sqrt(np.log(self.t) / self.N)
        return np.argmax(ucb_values)
      
    def update(self, action, reward):
        self.N[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.N[action]
        

# Running simulations
def run_experiment(bandit, agent, steps = 1000, runs = 2000):
    avg_rewards = np.zeros(steps)
    
    for _ in range(runs):
        env = Bandit(bandit.k)
        agent_instance = agent()
        rewards = []
        
        for t in range(steps):
            action = agent_instance.select_action()
            reward = env.get_reward(action)
            agent_instance.update(action, reward)
            rewards.append(reward)
            
        avg_rewards += np.array(rewards)
        
    return avg_rewards / runs
  
# experiment parameters
steps = 1000
runs = 2000

# Run the different algorithms
rewards_greedy = run_experiment(Bandit(10), lambda: Greedy(k = 10), steps, runs)
rewards_epsilon_greedy = run_experiment(Bandit(10), lambda: EpsilonGreedy(k = 10, epsilon = 0.1), steps, runs)
rewards_gradient_bandit = run_experiment(Bandit(10), lambda: GradientBandit(k = 10, alpha = 0.1), steps, runs)
rewards_ucb = run_experiment(Bandit(10), lambda: UCB(k = 10, c = 2), steps, runs)

# plot results
plt.figure(figsize = (10, 5))
plt.plot(rewards_gradient_bandit, label = 'Gradient Bandit (a = 0.1)')
plt.plot(rewards_ucb, label = 'UCB (c = 2)')
plt.plot(rewards_epsilon_greedy, label = 'Epsilon-Greedy (e = 0.1)')
plt.plot(rewards_greedy, label = 'Greedy')
plt.xlabel('steps')
plt.ylabel('average reward')
plt.title('Comparison of Greedy, Epsilon-Greedy, Gradient Bandit, and UCB')
plt.legend()
plt.show()


###
# Parameter Study : EpsilonGreedy

epsilons = [0, 0.01, 0.1, 0.3, 0.5]
plt.figure(figsize = (10, 5))
for epsilon in epsilons:
    rewards = run_experiment(Bandit(10), lambda: EpsilonGreedy(k = 10, epsilon = epsilon), steps, runs)
    plt.plot(rewards, label = f' epsilon = {epsilon}')
plt.xlabel('steps')
plt.ylabel('average reward')
plt.title('Epsilon-Greedy performance for different values of the epsilon parameter')
plt.legend()
plt.show()


# Parameter Study: Greedy and Optimistic Initialization
q0_values = [0, 2, 5]
plt.figure(figsize = (10, 5))
for Q0  in q0_values:
    rewards = run_experiment(Bandit(10), lambda: EpsilonGreedy(k = 10, epsilon = 0, Q0 = Q0), steps, runs)
    plt.plot(rewards, label = f'Q0 = {Q0}')
plt.xlabel('steps')
plt.ylabel('average reward')
plt.title('Greedy performance for different initial Q0 values')
plt.legend()
plt.show()


# Parameter Study: Gradient Bandit
alphas = [0.01, 0.1, 0.4]
plt.figure(figsize = (10, 5))
for alpha in alphas:
    rewards = run_experiment(Bandit(10), lambda: GradientBandit(k = 10, alpha = alpha), steps, runs)
    plt.plot(rewards, label = f'alpha = {alpha}')
plt.xlabel('steps')
plt.ylabel('average reward')
plt.title('Gradient Bandit peformance for different values of the alpha parameter')
plt.legend()
plt.show()


# Parameter Study: UCB
c_values = [0.1, 1, 2, 5]
plt.figure(figsize = (10, 5))
for c in c_values:
    rewards = run_experiment(Bandit(10), lambda: UCB(k = 10, c = c), steps, runs)
    plt.plot(rewards, label = f'c = {c}')
plt.xlabel('steps')
plt.ylabel('average reward')
plt.title('UCB performance for different values of the c parameter')
plt.legend()
plt.show()