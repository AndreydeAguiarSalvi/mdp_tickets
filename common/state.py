from collections import namedtuple

import torch


class State:

    """
    Initialize the State. If masks is a str and environment is None, so
    a State will be created from str to be index in a q_table.
        * masks - list of torch.Tensor or a string (just for recreate q_table)
        * environment - a list of dict with possible actions.
        * last_reward - the reward achieved to create this state.
        * n_prunes - a int of the times that model was already pruned.
    """
    def __init__(self, masks, environment = None, last_reward = None, n_prunes = 0):
        if isinstance(masks, str) and environment is None:
            splits = masks.split('=')
            self.n_prunes = int(splits[1].split(',')[0])
            self.remaining_weights = float(splits[2].split('\n')[0])
            self.last_reward = float(splits[3].split('\n')[0])
        else:
            self.masks = masks
            self.environment = environment
            self.last_reward = last_reward
            self.n_prunes = n_prunes
            self.remaining_weights = 0
            for mask in self.masks:
                self.remaining_weights += torch.sum(mask.weight)

    
    # def __hash__(self):
    #     return (hash(self.n_prunes) ^ hash(self.remaining_weights))
    
    def __hash__(self):
        return hash(str(self.n_prunes) + str(self.remaining_weights))

    def __eq__(self, other):
        return isinstance(other, State) and self.n_prunes == other.n_prunes and self.remaining_weights == other.remaining_weights

    def __str__(self):
        return f"State(\n\tn_prunes={self.n_prunes},\n\tremaining_weights={self.remaining_weights}\n\tlast_reward={self.last_reward}\n)"
    

    """
        Increments the number of prunes
    """
    def prune(self):
        self.n_prunes += 1