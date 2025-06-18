import matplotlib.pyplot as plt
import numpy as np

arrow_dict = {
    0: "↑", 1: "→", 2: "↓", 3: "←"
}

def plot_value_heatmap(V, env, title="State Value Function"):
    grid = V.reshape((env.size, env.size))
    plt.figure(figsize = (5,4))
    plt.imshow(grid, cmap = "viridis", interpolation = "nearest")
    plt.colorbar(label="value")
    plt.title(title)
    plt.xticks([]); plt.yticks([])
    for i in range(env.size):
        for j in range(env.size):
            plt.text(j, i, f"{grid[i, j]:.1f}", ha='center', va='center', color='white')
    plt.show()


def plot_policy_arrows(policy, env, title="Policy"):
    grid = np.array([arrow_dict[a] for a in policy]).reshape((env.size, env.size))
    plt.figure(figsize=(5,4))
    plt.table(cellText=grid, loc='center', cellLoc='center')
    plt.axis('off')
    plt.title(title)
    plt.show()
    