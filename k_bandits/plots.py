# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 14:10:23 2025

@author: vscerra
"""
import matplotlib.pyplot as plt

def plot_results(results_dict, title = "Performance Comparison", ylabel = "Average Reward"):
    """Plots the reward curves for different algorithms."""
    plt.figure(figsize = (10, 6))
    for label, rewards in results_dict.items():
        plt.plot(rewards, label = label)
    plt.xlabel("Steps")
    plt.ylabel(ylabel)
    plt.legend()
    plt.title(title)
    plt.show()