import numpy as np

def one_step_lookahead(state, V, P, gamma):
    """
    Computes action-values for all actions from a given state.
    """
    A = np.zeros(len(P[state]))
    for a in range(len(P[state])):
        for prob, next_s, reward in P[state][a]:
            A[a] += prob * (reward + gamma * V[next_s])
    return A

def value_iteration(env, theta=1e-4):
    V = np.zeros(env.n_states)
    P = env.get_transition_model()
    gamma = env.gamma

    while True:
        delta = 0
        for s in range(env.n_states):
            A = one_step_lookahead(s, V, P, gamma)
            best_action_value = np.max(A)
            delta = max(delta, abs(best_action_value - V[s]))
            V[s] = best_action_value
        if delta < theta:
            break

    # Derive deterministic policy from value function
    policy = np.zeros(env.n_states, dtype=int)
    for s in range(env.n_states):
        A = one_step_lookahead(s, V, P, gamma)
        policy[s] = np.argmax(A)
    return V, policy

