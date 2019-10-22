import torch

import random
import numpy as np

from copy import deepcopy

from pytorch.net import NN
from pytorch.utils import validation, q_value

from pytorch.state import State

class Agent:

    """
        Initialize the Agent. 
        * initial_environment - dict with all possible prune positions.
        * reward_computation - How reward will be computed.
            - ACCURACY - the reward will be equal to accuracy
            - LOSS - the reward will be the negative loss
            - RcRa - the reward computing the compression and accuracy
        * prune_percentage float - Percent of weights to be pruned
        * model NN - The neural network to be pruned
        * validloader DataLoader - The loader of the validation.
        * criterion Criteria - The criterion of computing the loss.
        * device - str with the GPU
    """
    def __init__(self, config, initial_environment, model, valid_loader, criterion):
        self.initial_environment = initial_environment
        self.reward_computation = config['agent']['reward_type']
        self.prune_percentage = config['agent']['prune_percentage']
        self.model = model
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.device = config['device']
        self.epsilon = config['agent']['epsilon']
        self.is_done = False
        self.is_gameover = config['agent']['is_gameover']
        self.gameover_threshold = config['agent']['gameover_threshold']
        self.total_weights = 0
        for mask in model.masks:
            self.total_weights += np.prod(mask.weight.shape, 0)
        self.remaining_weights = (1. - self.prune_percentage) * self.total_weights
    

    """
        This method make the agent prune some row or column from the NN.
        * state - list(torch.Tensor) with the masks of the NN.
        * action - dict(key, value) from dictionary of actions
    """
    def act(self, state, action):
        # decoding the numpy.float action to {key, value} action
        action = state.environment[action]
        new_masks = self.__new_pruned_mask(state, action)
        
        # validate the model
        self.model.set_new_masks(new_masks)
        if not self.model.masks[0].weight.is_cuda:
            self.model.masks = self.model.masks.to(self.device)
        valid_loss, accuracy = validation(self.model, self.valid_loader, self.criterion)

        # compute the reward
        reward = self.__compute_reward(state, valid_loss, accuracy)
        self.model.masks[0] = self.model.masks[0].to('cpu')
        self.model.masks[1] = self.model.masks[1].to('cpu')
        self.model.masks[2] = self.model.masks[2].to('cpu')
        
        # new subenvironment
        new_env = deepcopy(state.environment)
        new_env.remove(action)
        
        new_state = State(new_masks, new_env, reward, state.n_prunes)
        new_state.prune()

        # if self.is_gameover:
        #     # Verify if the game has game over
        #     if reward == -100:
        #         self.is_done = True
        
        # Verify if the game has winned
        if (new_state.remaining_weights <= self.remaining_weights):
            self.is_done = True
            reward *= 10
        
        return new_state, reward, self.is_done


    """
        Choose an action with epsilon percent of randomness
    """
    def choose_action(self, q_table, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, len(state.environment)-1)
        return np.argmax(q_value(q_table, state))
    

    def __new_pruned_mask(self, state, action):
        masks = deepcopy(state.masks)
        """
            Unlike the cart example, here we do not have unpossible actions, as i.e.
            the car jump out the table. The unique restriction is do not prune the 
            same connections, but we will drop the already performed action from the
            possible actions (state.environment).
        """
        
        # finding the index to prune
        key = str(action.keys())
        values = list(action.values())
        with torch.no_grad():
            layer = values[0][0]
            if 'row' in key:
                row = values[0][1]
                masks[layer].weight[row, :] = .0
                masks[layer+1].weight[:, row] = .0
            else:
                raise RuntimeError('{} -> No recognized key'.format(key))

        return masks
    

    def __compute_reward(self, state, valid_loss, accuracy):
        if self.is_gameover:    
            if (self.reward_computation == 'ACCURACY'):
                return accuracy if accuracy > self.gameover_threshold * state.last_reward else -100
            elif (self.reward_computation == 'LOSS'):
                return -valid_loss if valid_loss < ( (1. - self.gameover_threshold) + 1.) * state.last_reward else -100
            elif (self.reward_computation == 'RcRa'):
                rw = (accuracy) * (state.remaining_weights / self.total_weights)
                return rw if rw > self.gameover_threshold * state.last_reward else -100
        else:
            if (self.reward_computation == 'ACCURACY'):
                return accuracy
            elif (self.reward_computation == 'LOSS'):
                return -valid_loss
            elif (self.reward_computation == 'RcRa'):
                return (accuracy) * (self.total_weights / state.remaining_weights)