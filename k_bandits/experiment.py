# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 14:08:12 2025

@author: vscerra
"""
import numpy as np
from k_bandits.bandit import Bandit

def run_experiment(algorithm_class, algorithm_params = {}, k=10, steps=1000, runs=2000):
    """Runs multiple trials of a bandit algorithm and returns average rewards."""
    avg_rewards = np.zeros(steps)
    optimal_action_counts = np.zeros(steps)

    for _ in range(runs):
        env = Bandit(k)
        agent = algorithm_class(k, **algorithm_params) #pass parameters from input
        rewards = []
        optimal_action = np.argmax(env.q_true) # finding optimal action

        for t in range(steps):
            action = agent.select_action()
            reward = env.get_reward(action)
            agent.update(action, reward)
            rewards.append(reward)
            optimal_action_counts[t] += (action == optimal_action) # tracking optimal actions

        avg_rewards += np.array(rewards)
        
        

    return avg_rewards / runs, optimal_action_counts / runs # normalize for percentage