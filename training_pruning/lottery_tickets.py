import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

import os
import sys
import json
import math
import time
import logging
import numpy as np
import pandas as pd

sys.path.append('../')
from common.net import NN
from common.utils import q_value, train, get_optimizer

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
    test_data = datasets.MNIST(root='../data', train=False,
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
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
        num_workers=num_workers)

    return train_loader, valid_loader, test_loader


def __prune(model, config):

    for (or_value, mask_value) in zip(model.layers, model.masks):
        
        # Number of neurons to be prunned
        n_pruned_neurons = math.floor(torch.sum(mask_value.weight) * config['prune']['rate'])
        # Getting all the available values to possibly be pruned
        valid_values = torch.masked_select(or_value.weight, mask_value.weight.byte())
        # Getting the values to be prunned.
        smallest_values = torch.topk(input=torch.abs(valid_values), k=n_pruned_neurons, largest=False)
        # Getting the higher and smaller valid values to be pruned.
        # All non-zero elements smaller than higher_of_smallest and 
        # higher than smaller_of_smallest will be pruned
        higher_of_smallest = smallest_values.values[len(smallest_values.values)-1]
        smaller_of_smallest = smallest_values.values[0]
        # get the indexes where prune
        indexes_to_prune = torch.nonzero( (smaller_of_smallest <= torch.abs(or_value.weight)) & (torch.abs(or_value.weight) <= higher_of_smallest) )
        
        # Prunning the values
        with torch.no_grad():
            for index_to_prune in indexes_to_prune:
                or_value.weight[index_to_prune[0]] [index_to_prune[1]] = .0
                mask_value.weight[index_to_prune[0]] [index_to_prune[1]] = .0  

    return model


def __verify_stop(model, config):
    remaining_weights = .0
    total_weights = .0
    for mask in model.masks:
        remaining_weights += torch.sum(mask.weight)
        total_weights += np.prod(mask.weight.shape, 0)

    if remaining_weights <= total_weights * (1. - config['prune']['compression']):
        return True
    
    return False


def main():
    logging.basicConfig(level=logging.DEBUG,
                        format='[%(asctime)s %(filename)s] %(message)s')
    json_file = open('params.json')
    json_str = json_file.read()
    config = json.loads(json_str)


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
            logging.info('Pruning the model')
            model = __prune(model, config)
            
            if config['prune']['reward_weights'] == True:
                logging.info('Rewarding the weights')
                model.reward()
        
            done = __verify_stop(model, config)
        
        epoch += 1

    ##############
    # Last Train #
    ##############
    train(model, train_loader, valid_loader, criterion, optimizer, 5, config['train']['print_every'], config['device'])
    
    checkpoint = config['policy_dir'] + '/Train-{}-epochs__Prune-each-{}__Prune-rate-{}.pt'.format(
            config['train']['epochs'], config['prune']['each'], config['prune']['rate']
    )
    torch.save(model.state_dict(), checkpoint)
    logging.info("Model checkpoint saved to %s" % checkpoint)

    ######################
    # Validate the model #
    ######################
    test_loss = 0.0
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
        config['policy_dir'] + '/Accuracy__Lottery-train-{}-epochs__Prune-each-{}__Prune-rate-{}.tsv'.format(
            config['train']['epochs'], config['prune']['each'], config['prune']['rate']
        ), 
        index = False, 
        header = ['Class', 'Accuracy', 'Right_Instances', 'Total_Instances'],
        sep = '\t'
    )


if __name__ == "__main__":
    main()