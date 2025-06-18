import numpy as np

def one_step_lookahead(state, V, env):
    """
    Computes action-values for all actions from a given state.
    """
    action_values = np.zeros(len(env.get_actions()))
    for a in env.get_actions():
        total = 0
        for _ in range(10):  # Monte Carlo sampling to estimate transitions
            env.state = state
            next_state, reward, _ = env.step(a)
            total += reward + env.gamma * V[next_state]
        action_values[a] = total / 10
    return action_values


def value_iteration(env, theta=1e-4):
    V = np.zeros(env.n_states)
    while True:
        delta = 0
        for s in range(env.n_states):
            env.state = s
            A = one_step_lookahead(s, V, env)
            best_action_value = np.max(A)
            delta = max(delta, abs(best_action_value - V[s]))
            V[s] = best_action_value
        if delta < theta:
            break

    # Derive deterministic policy from value function
    policy = np.zeros(env.n_states, dtype=int)
    for s in range(env.n_states):
        env.state = s
        A = one_step_lookahead(s, V, env)
        policy[s] = np.argmax(A)
    return V, policy

