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
import argparse
import numpy as np
import pandas as pd

sys.path.append('../')
from common.net import NN
from common.utils import get_optimizer


def __load_data(config):
    # number of subprocesses to use for data loading
    num_workers = 0
    # how many samples per batch to load
    batch_size = config['train']['train_batch']
    # percentage of training set to use as validation
    valid_size = config['train']['valid_batch']

    # convert data to torch.FloatTensor
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])


    # choose the training and test datasets
    train_data = datasets.CIFAR10(root='../data', train=True,
                                    download=False, transform=train_transform)
    test_data = datasets.CIFAR10(root='../data', train=False,
                                    download=False, transform=test_transform)

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
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    return train_loader, valid_loader, test_loader


def __create_args():
    # Arguments
    parser = argparse.ArgumentParser(
        description='Reuse the mask from another train.'
    )
    # General parameters
    parser.add_argument('-d', '--device', help='Cuda to run the model', type=str, default=None)
    
    # Model parameters
    parser.add_argument('-m_a', '--model_architecture', help='The number of neurons of the NN', type=list, default=None)
    
    # Train parameters
    parser.add_argument('-e', '--epochs', help='The number of epochs to train the model before the pruning', type=int, required=True)
    parser.add_argument('-prt', '--print_every', help='Print accuracy at every p_e epochs', type=int, default=None)

    parser.add_argument('-m_p', '--model_path', help='The path to load model and reuse the mask', type=str, required=True)

    args = vars(parser.parse_args())
    return args


def __adjust_config(config, args):
    if args['device'] is not None:
        config['device'] = args['device']
    
    if args['model_architecture'] is not None:
        config['model']['architecture'] = args['model_architecture']
    
    if args['epochs'] is not None:
        config['epochs'] = args['epochs']
    if args['print_every'] is not None:
        config['train']['print_every'] = args['print_every']

    if args['model_path'] is not None:
        config['model_path'] = args['model_path']
    
    return config


def main():
    logging.basicConfig(level=logging.DEBUG,
                        format='[%(asctime)s %(filename)s] %(message)s')
    json_file = open('params.json')
    json_str = json_file.read()
    config = json.loads(json_str)

    args = __create_args()
    config = __adjust_config(config, args)

    # loading the dataset
    train_loader, valid_loader, test_loader = __load_data(config)

    # Creating the model
    model = NN(config['model']['architecture'], is_maskable = True)
    model = model.to(config['device'])
    # Reusing the mask
    for (_, _, files) in os.walk(config['model_path']):
        file = [s for s in files if 'pt' in s]
    reused = NN(config['model']['architecture'], is_maskable = True)
    checkpoint = torch.load(config['model_path'] + '/' + file[0], map_location=config['device'])
    logging.info('Loading checkpoint: {}'.format(config['model_path'] + '/' + file[0]))
    reused.load_state_dict(checkpoint)

    model.masks = reused.masks
    model.masks = model.masks.to(config['device'])
    total_weights = 0
    remaining_weights = 0
    for mask in model.masks:
        total_weights += np.prod(mask.weight.shape, 0)
        remaining_weights += torch.sum(mask.weight)
    logging.info('Total of weights: {}\tRemaining weights: {}'.format(total_weights, remaining_weights))
    
    # Getting the criterion, optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(config, model)
    
    
    for e in range(config['epochs']):
        #########
        # Train #
        #########
        model.train()
        train_loss = .0
        train_correct = .0
        train_total = .0
        for data, target in train_loader:
            data, target = data.to(config['device']), target.to(config['device'])

            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            # update test loss 
            train_loss += loss.item()*data.size(0)
            # convert output probabilities to predicted class
            _, pred = torch.max(output, 1)
            # compare predictions to true label
            correct = np.squeeze(pred.eq(target.data.view_as(pred)))
            # calculate train accuracy for each object class
            train_correct += np.sum(correct)
            train_total += data.shape[0]

        ############
        # Validate #
        ############
        model.eval()
        valid_loss = .0
        valid_correct = .0
        valid_total = .0
        for data, target in valid_loader:
            data, target = data.to(config['device']), target.to(config['device'])

            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            loss = criterion(output, target)
            # update valid loss 
            valid_loss += loss.item()*data.size(0)
            # convert output probabilities to predicted class
            _, pred = torch.max(output, 1)
            # compare predictions to true label
            correct = np.squeeze(pred.eq(target.data.view_as(pred)))
            # calculate valid accuracy for each object class
            valid_correct += np.sum(correct)
            valid_total += data.shape[0]
        
        logging.info('Train Loss: {} Train Accuracy: {}\tValid Loss: {} Valid Accuracy: {}'.format(
            train_loss, train_correct/train_total,
            valid_loss, valid_correct/valid_total
        ))

    checkpoint = config['model_path'] + 'model-on-CIFAR10__optimizer-{}__train-{}.pth'.format(
        config['optimizer']['type'],
        config['epochs']
    )
    torch.save(model.state_dict(), checkpoint)
    logging.info("Model checkpoint saved to %s" % checkpoint)


    ######################
    # Evaluate the model #
    ######################
    test_loss = 0.0
    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    model.eval() # prep model for evaluation
    for data, target in test_loader:
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
    test_loss = test_loss/len(test_loader.sampler)
    logging.info('Test Loss: {:.6f}\n'.format(test_loss))

    results = []
    for i in range(10):
        if class_total[i] > 0:
            logging.info('Test Accuracy of %5s: %.5f%% (%2d/%2d)' % (
                classes[i], 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
            results.append([classes[i], 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])])
        else:
            logging.info('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

    logging.info('Test Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))

    results = pd.DataFrame(results)
    results.to_csv(
        config['model_path'] + 'model-on-CIFAR10__Accuracy__Model-train-{}.tsv'.format(
            config['epochs'], 
        ), 
        index = False, 
        header = ['Class', 'Accuracy', 'Right_Instances', 'Total_Instances'],
        sep = '\t'
    )


if __name__ == "__main__":
    main()