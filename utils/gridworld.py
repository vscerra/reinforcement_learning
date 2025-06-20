import numpy as np

class Gridworld:
    def __init__(self, size=4, start=0, goal=15, traps=None, gamma=0.9):
        self.size = size
        self.n_states = size * size
        self.start = start
        self.goal = goal if goal is not None else self.n_states - 1
        self.traps = traps if traps else []
        self.gamma = gamma
        self.reset()

    def state_to_pos(self, s):
        return (s // self.size, s % self.size)
    
    def pos_to_state(self, pos):
        return pos[0] * self.size + pos[1]
    
    def reset(self):
        self.state = self.start
        return self.state
    
    def get_actions(self):
        return [0, 1, 2, 3] # 0-up, 1-right, 2-down, 3-left
    
    def step(self, action):
        row, col = self.state_to_pos(self.state)

        moves = {
            0: (-1, 0), #up
            1: (0, 1), #right
            2: (1, 0), #down
            3: (0, -1) #left
        }

        # Apply action with 80% probability, otherwise it's a random move
        if np.random.rand() < 0.8:
            dr, dc = moves[action]
        else:
            dr, dc = moves[np.random.choice(self.get_actions())]

        new_row = np.clip(row + dr, 0, self.size - 1)
        new_col = np.clip(col + dc, 0, self.size - 1)
        next_state = self.pos_to_state((new_row, new_col))

        reward = -1
        done = False
        
        if next_state == self.goal:
            reward = 10
            done = True
        elif next_state in self.traps:
            reward = -10
            done = True

        self.state = next_state
        return next_state, reward, done
    
    def get_transition_model(self):
        """ 
        Returns a dict[state][action] = list of (prob, next_state, reward)
        """
        model = {}
        for s in range(self.n_states):
            model[s] = {}
            row, col = self.state_to_pos(s)
            for a in self.get_actions():
                outcomes = []
                for action_option, prob in zip([a, (a+1)%4, (a+3)%4], [0.8, 0.1, 0.1]):
                    dr, dc = {0:(-1,0), 1:(0,1), 2:(1,0), 3:(0,-1)}[action_option]
                    new_row = np.clip(row + dr, 0, self.size - 1)
                    new_col = np.clip(col + dc, 0, self.size - 1)
                    s_next = self.pos_to_state((new_row, new_col))

                    if s_next == self.goal:
                        reward = 10
                        done = True
                    elif s_next in self.traps:
                        reward = -10
                        done = True
                    else:
                        reward = -1
                        done = False
                    
                    outcomes.append((prob, s_next, reward))
                model[s][a] = outcomes
        return model
    