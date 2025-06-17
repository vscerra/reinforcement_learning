# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 14:12:17 2025

@author: vscerra
"""
import matplotlib.pyplot as plt
from algorithms import Greedy, EpsilonGreedy, UCB, GradientBandit
from bandit_experiment import run_experiment
from bandit_plots import plot_results

# Experiment settings
steps = 1000
runs = 2000
k = 10  # Number of arms

# Run Greedy
rewards = run_experiment(lambda: Greedy(k = k), steps, runs)
plot_results(rewards, label = "Greedy")


# Run Epsilon-Greedy with different epsilon values
epsilons = [0, 0.01, 0.1, 0.3, 0.5]
plt.figure(figsize = (10, 5))

for epsilon in epsilons:
    rewards = run_experiment(lambda: EpsilonGreedy(k=k, epsilon=epsilon), k, steps, runs)
    plt.plot(rewards, label=f"ε={epsilon}")
    
plt.xlabel('steps')
plt.ylabel('average reward')
plt.title('Epsilon-Greedy: Parameter search of epsilon values')
plt.legend()
plt.show()


# Run UCB with different confidence parameters
c_values = [0.1, 1, 2, 5]
plt.figure(figsize = (10, 5))

for c in c_values:
    rewards = run_experiment(lambda: UCB(k=k, c=c), k, steps, runs)
    plt.plot(rewards, label=f"c={c}")
    
plt.xlabel('steps')
plt.ylabel('average reward')
plt.title('UCB: Parameter search of c values')
plt.legend()
plt.show()    

# Run Gradient Bandit with different alpha values
alphas = [0.01, 0.1, 0.4]
plt.figure(figsize = (10, 5))

for alpha in alphas:
    rewards = run_experiment(lambda: GradientBandit(k=k, alpha=alpha), k, steps, runs)
    plt.plot(rewards, label=f"α={alpha}")

plt.xlabel('steps')
plt.ylabel('average reward')
plt.title('Gradient Bandit: Parameter search of alpha values')
plt.legend()
plt.show()