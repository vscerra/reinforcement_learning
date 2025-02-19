# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 14:08:12 2025

@author: vscerra
"""
import numpy as np
from k_bandits.bandit import Bandit

def run_experiment(algorithm, k=10, steps=1000, runs=2000):
    """Runs multiple trials of a bandit algorithm and returns average rewards."""
    avg_rewards = np.zeros(steps)

    for _ in range(runs):
        env = Bandit(k)
        agent = algorithm()
        rewards = []

        for t in range(steps):
            action = agent.select_action()
            reward = env.get_reward(action)
            agent.update(action, reward)
            rewards.append(reward)

        avg_rewards += np.array(rewards)

    return avg_rewards / runs