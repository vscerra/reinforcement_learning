# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 14:10:23 2025

@author: vscerra
"""
import matplotlib.pyplot as plt

def plot_results(rewards, label):
    """Plots the reward curves for different algorithms."""
    plt.plot(rewards, label=label)
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.title("Performance Comparison")
    plt.show()