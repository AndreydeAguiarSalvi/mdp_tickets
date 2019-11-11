import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

import os
import sys
import json
import time
import logging
import numpy as np
import pandas as pd

sys.path.append('../')
from common.agent import Agent
from common.state import State
from common.net import NN
from common.utils import q_value, train, get_optimizer, policy_loader

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
    train_data = datasets.MNIST(root='../data', train=True,
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


def __adjust_config(config):
    x = config['policy_dir'].split('__')
    environment_protocol = x[2].split('-')[1]
    prune_percentage = float(x[3].split('-')[1])

    config['environment_protocol'] = environment_protocol
    config['prune_percentage'] = prune_percentage

    sub_working_dir = '{}/models/{}'

    return config


def __prune(model, action):

    masks = model.masks
    
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

    return model


def main():
    logging.basicConfig(level=logging.DEBUG,
                        format='[%(asctime)s %(filename)s] %(message)s')
    json_file = open('params.json')
    json_str = json_file.read()
    config = json.loads(json_str)

    config = __adjust_config(config)

    if len(sys.argv) == 2:
        config['policy_dir'] = sys.argv[1]

    # loading the dataset
    train_loader, valid_loader, test_loader = __load_data(config)

    # Creating the model
    model = NN(config['model']['architecture'], is_maskable = True)
    model = model.to(config['device'])

    # Getting the criterion, optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(config, model)
    
    policy = policy_loader(config['policy_dir'] + '/actions_to_prune.tsv')

    ###################
    # Train and Prune #
    ###################
    done = False
    epoch = 0
    while not done:
        logging.info('')
        logging.info('Global step {}'.format(epoch))
        logging.info('')
        train(model, train_loader, valid_loader, criterion, optimizer, config['train']['epochs'], config['train']['print_every'], config['device'])

        if epoch % config['prune']['each'] == 0 and epoch != 0:

            action = policy[0]
            model = __prune(model, action)
            logging.info('Action performed: {}'.format(action))
            policy.pop(0)
            if len(policy) == 0:
                done = True

            if config['prune']['reward_weights'] == True:
                logging.info('Rewarding the weights')
                model.reward()
            
            model.masks = model.masks.to(config['device'])
        
        epoch += 1

    ##############
    # Last Train #
    ##############
    train(model, train_loader, valid_loader, criterion, optimizer, 5, config['train']['print_every'], config['device'])
    
    checkpoint = config['policy_dir'] + '/model__prune_each-{}__optimizer-{}.pth'.format(
        config['train']['epochs'] * config['prune']['each'],
        config['optimizer']['type']
    )
    torch.save(model.state_dict(), checkpoint)
    logging.info("Model checkpoint saved to %s" % checkpoint)

    ######################
    # Validate the model #
    ######################
    test_loss = 0.0
    classes = range(0, 10)
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    model.eval() # prep model for evaluation

    for data, target in valid_loader:
        data, target = data.to(config['device']), target.to(config['device'])

        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update test loss 
        test_loss += loss.item()*data.size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # compare predictions to true label
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))
        # calculate test accuracy for each object class
        for i in range(len(target)):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    # calculate and print avg test loss
    test_loss = test_loss/len(valid_loader.sampler)
    logging.info('Valid Loss: {:.6f}\n'.format(test_loss))

    results = []
    for i in range(10):
        if class_total[i] > 0:
            logging.info('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                str(i), 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
            results.append([str(i), 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])])
        else:
            logging.info('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

    logging.info('Test Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))

    results = pd.DataFrame(results)
    results.to_csv(
        config['policy_dir'] + '/Accuracy__Model-train-{}-epochs__Prune-each-{}.tsv'.format(
            config['train']['epochs'], config['prune']['each']
        ), 
        index = False, 
        header = ['Class', 'Accuracy', 'Right_Instances', 'Total_Instances'],
        sep = '\t'
    )


if __name__ == "__main__":
    main()