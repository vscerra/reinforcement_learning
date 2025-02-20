# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 14:10:23 2025

@author: vscerra
"""
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_results(results_dict, title = "Performance Comparison", ylabel = "Average Reward"):
    """Plots the reward curves for different algorithms."""
    
    plt.figure(figsize = (10, 6))
    
    # set colormap
    colormap = cm.get_cmap('plasma', len(results_dict))
    
    for i, (label, rewards) in enumerate(results_dict.items()):
        plt.plot(rewards, label = label, color = colormap(i))
        
        
    plt.xlabel("Steps")
    plt.ylabel(ylabel)
    plt.legend()
    plt.title(title)
    plt.show()