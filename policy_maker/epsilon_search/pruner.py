import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

import os
import sys
import json
import time
import random
import logging
import argparse
import numpy as np
import pandas as pd
from copy import deepcopy
from random import shuffle

sys.path.append('../../')
from common.agent import Agent
from common.state import State
from common.net import NN
from common.utils import q_value, create_environment, train, validation, get_optimizer, q_table_loader, q_table_saver


def __load_data(config):
    # number of subprocesses to use for data loading
    num_workers = 0
    # how many samples per batch to load
    batch_size = config['train']['train_batch']
    # percentage of training set to use as validation
    valid_size = config['train']['valid_batch']

    # convert data to torch.FloatTensor
    transform = transforms.ToTensor()

    # choose the training and test datasets
    train_data = datasets.MNIST(root=config['working_dir'] + '/data', train=True,
                                    download=False, transform=transform)

    # obtain training indices that will be used for validation
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # prepare data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
        sampler=train_sampler, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
        sampler=valid_sampler, num_workers=num_workers)

    return train_loader, valid_loader


def __create_args():
    # Arguments
    parser = argparse.ArgumentParser(
        description='Performs a Alpha Search with MDP.'
    )
    # General parameters
    parser.add_argument('-d', '--device', help='Cuda to run the model', type=str, default=None)
    parser.add_argument('-e_p', '--environment_protocol', help='The type of pruning', type=str, default=None)
    # Model parameters
    parser.add_argument('-m_a', '--model_architecture', help='The number of neurons of the NN', type=list, default=None)
    # MDP parameters
    parser.add_argument('-N_E', '--N_EPISODES', help='The maximum of episodes to agent perform', type=int, default=None)
    parser.add_argument('-M_S', '--MAX_STEPS_PER_EPISODES', help='Maximum of steps per episode', type=int, default=None)
    parser.add_argument('-A', '--ALPHA', help='Minimum number of Alpha for Alpha Search', type=float, default=None)
    parser.add_argument('-G', '--GAMMA', help='Gamma', type=float, default=None)
    parser.add_argument('-Q_C', '--Q_COMPUTATION', help='The equation to compute the Quality', type=str, default=None)
    # Agent parameters
    parser.add_argument('-p_e', '--prune_percentage', help='Percentage of weights to prune', type=float, default=None)
    parser.add_argument('-eps', '--epsilon', help='Percentage of random actions from agent', type=float, default=None)
    parser.add_argument('-r_t', '--reward_type', help='Type of reward computation', type=str, default=None)
    # Train parameters
    parser.add_argument('-e', '--epochs', help='The number of epochs to train the model before the pruning', type=int, default=None)
    parser.add_argument('-prt', '--print_every', help='Print accuracy at every p_e epochs', type=int, default=None)

    args = vars(parser.parse_args())
    return args


def __adjust_config(args, config):
    if args['device'] is not None:
        config['device'] = args['device']
    if args['environment_protocol'] is not None:
        config['environment_protocol'] = args['environment_protocol']
    if args['model_architecture'] is not None:
        config['model']['architecture'] = args['model_architecture']
    
    if args['N_EPISODES'] is not None:
        config['mdp']['N_EPISODES'] = args['N_EPISODES']
    if args['MAX_STEPS_PER_EPISODES'] is not None:
        config['mdp']['MAX_STEPS_PER_EPISODES'] = args['MAX_STEPS_PER_EPISODES']
    if args['ALPHA'] is not None:
        config['mdp']['ALPHA'] = args['ALPHA']
    if args['Q_COMPUTATION'] is not None:
        config['mdp']['Q_COMPUTATION'] = args['Q_COMPUTATION']

    if args['prune_percentage'] is not None:
        config['agent']['prune_percentage'] = args['prune_percentage']
    if args['epsilon'] is not None:
        config['agent']['epsilon'] = args['epsilon']
    if args['reward_type'] is not None:
        config['agent']['reward_type'] = args['reward_type']
    
    if args['epochs'] is not None:
        config['train']['epochs'] = args['epochs']
    if args['print_every'] is not None:
        config['train']['print_every'] = args['print_every']
    
    return config


def main():
    logging.basicConfig(level=logging.DEBUG,
                        format="[%(asctime)s %(filename)s] %(message)s")
    json_file = open('params.json')
    json_str = json_file.read()
    config = json.loads(json_str)

    args = __create_args()
    config = __adjust_config(args, config)

    # loading the dataset
    train_loader, valid_loader = __load_data(config)

    # Creating the model
    model = NN(config['model']['architecture'], is_maskable = True)
    model = model.to(config['device'])
    initial_mask = model.masks

    # Getting the criterion, optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(config, model)

    #########################
    # Agent and Environment #
    #########################
    ACTIONS = create_environment(model.masks, config['environment_protocol'])
    random.seed(42)
    shuffle(ACTIONS)
    N_STATES = len(ACTIONS)
    N_EPISODES = config['mdp']['N_EPISODES']
    MAX_EPISODE_PER_STEPS = config['mdp']['MAX_STEPS_PER_EPISODES']
    ALPHA = config['mdp']['ALPHA']
    GAMMA = config['mdp']['GAMMA']
    epsilons = np.linspace(1.0, config['agent']['epsilon'], N_EPISODES)

    q_table = dict()
    start_state = State(model.masks, ACTIONS)
    

    ##########################
    # Create sub_working_dir #
    ##########################
    sub_working_dir = '{}/results/{}/{}/{}/{}'.format(
        config['working_dir'],
        config['model']['name'], 
        '_' + config['mdp']['Q_COMPUTATION'],
        '{}_{}_{}/{}_{}'.format(
            time.strftime("%d", time.localtime()),
            time.strftime("%m", time.localtime()),
            time.strftime("%Y", time.localtime()),
            time.strftime("%H", time.localtime()),
            time.strftime("%M", time.localtime())
        ),
        'EPSILON_SEARCH__ALPHA-{}__GAMMA-{}__PRUNE_TYPE-{}__PRUNE_PERCENT-{}__MIN_EPSILON-{}__REWARD_TYPE-{}'.format(
            ALPHA, GAMMA if config['mdp']['Q_COMPUTATION'] != 'QL_M' else 'None',
            config['environment_protocol'], 
            config['agent']['prune_percentage'],
            config['agent']['epsilon'],
            config['agent']['reward_type']
        )
    )

    if not os.path.exists(sub_working_dir):
        os.makedirs(sub_working_dir)
    config["sub_working_dir"] = sub_working_dir
    logging.info("sub working dir: %s" % sub_working_dir)

    ###############
    # Begin Train #
    ###############
    train(model, train_loader, valid_loader, criterion, optimizer, config['train']['epochs'], config['train']['print_every'], config['device'])
    loss, accuracy = validation(model, valid_loader, criterion)
    logging.info('Validation Loss performed: {}\tValidation Accuracy performed: {}'.format(loss, accuracy))

    if config['agent']['reward_type'] == 'ACCURACY':
        start_state.last_reward = -(1. - accuracy)
    elif config['agent']['reward_type'] == 'LOSS':
        start_state.last_reward = -loss
    elif config['agent']['reward_type'] == 'ACC_COMPRESSION':
        start_state.last_reward = -(1. - accuracy) 
    elif config['agent']['reward_type'] == 'MY_RCRA':
        start_state.last_reward = -(1. - accuracy)

    #########
    # Prune #
    #########
    for e in range(N_EPISODES):

        state = deepcopy(start_state)
        total_reward = .0
        
        config['agent']['epsilon'] = epsilons[e]
        agent = Agent(config, ACTIONS, model, valid_loader, criterion) 

        for i in range(MAX_EPISODE_PER_STEPS):
            action = agent.choose_action(q_table, state)
            
            next_state, reward, done = agent.act(state, action)
            total_reward += reward
            
            if config['mdp']['Q_COMPUTATION'] == 'QL_M':
                # Q-Learning from Ghallab, Nau and Traverso
                q_value(q_table, state)[action] = q_value(q_table, state, action) + \
                    ALPHA * (reward + np.max(q_value(q_table, next_state)) - q_value(q_table, state, action))

            elif config['mdp']['Q_COMPUTATION'] == 'QL_WIKI':
                # Q-Learning from from Wikipedia
                q_value(q_table, state)[action] = (1. - ALPHA) * q_value(q_table, state, action) + \
                    ALPHA * (reward + GAMMA * np.max(q_value(q_table, next_state)))

            del state
            state = next_state
            if done:
                break

        logging.info("Episode {}: reward type {}: total reward -> {}".format(e + 1, config['agent']['reward_type'], total_reward))

    #####################
    # Save the solution #
    #####################
    q_table_saver(q_table, config['sub_working_dir'], '/q_table.tsv')

    agent = Agent(config, ACTIONS, model, valid_loader, criterion)
    my_state = start_state
    result = []
    done = False
    while not done:
        sa = q_value(q_table, my_state)
        my_action = np.argmax(sa)
        action = my_state.environment[my_action]
        my_state, reward, done = agent.act(my_state, my_action)
        result.append([action, reward])

    final = pd.DataFrame(result, columns = ['Action', 'Reward'])
    final.to_csv(config['sub_working_dir'] + '/actions_to_prune.tsv', sep='\t', index=False)



if __name__ == "__main__":
    main()