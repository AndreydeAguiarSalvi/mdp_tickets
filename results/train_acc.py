import os
import sys
import json
import torch
import logging
import argparse

logging.basicConfig(level=logging.DEBUG,
                        format='[%(asctime)s %(filename)s] %(message)s')

import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

sys.path.append('../')

from common.net import NN
from torchvision import datasets

"""
    This script will walk on all subfolders of config['model_path'],
    will compute the accuracy of each class on the training dataset 
    and save on {something}_train_acc.tsv
"""

def __load_data(config):
    # number of subprocesses to use for data loading
    num_workers = 0
    # how many samples per batch to load
    batch_size = config['batch_size']
    valid_size = config['valid_size']

    # convert data to torch.FloatTensor
    transform = transforms.ToTensor()

    # choose the training and test datasets
    train_data = datasets.MNIST(root='../data', train=True,
                                    download=False, transform=transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)

    # prepare data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
        sampler=train_sampler, num_workers=num_workers)

    return train_loader


def __make_evaluation(model, loader, config, dt):
    classes = range(0, 10)
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    for images, labels in loader:
        images, labels = images.to(config['device']), labels.to(config['device'])
        
        output = model(images)
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # compare predictions to true label
        correct = np.squeeze(pred.eq(labels.data.view_as(pred)))
        # calculate test accuracy for each object class
        for i in range(len(labels)):
            label = labels.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    for i in range(10):
        if class_total[i] > 0:
            logging.info(
                    'Train Accuracy of %5s: %2d%% (%2d/%2d)' % (
                        str(i), 100 * class_correct[i] / class_total[i],
                        np.sum(class_correct[i]), np.sum(class_total[i])
                    )
                )
            dt.append([
                    'Train Accuracy of',
                        str(i), 100 * class_correct[i] / class_total[i],
                        np.sum(class_correct[i]), np.sum(class_total[i])
                    ]
                )
        else:
            logging.info('Train Accuracy of %5s: N/A (no training examples)' % (classes[i]))

    logging.info(
        'Train Accuracy (Overall): %2d%% (%2d/%2d)' % (
                100. * np.sum(class_correct) / np.sum(class_total),
                np.sum(class_correct), np.sum(class_total)
            )
        )
    dt.append([
        'Train Accuracy (Overall):',
            100. * np.sum(class_correct) / np.sum(class_total),
            np.sum(class_correct), np.sum(class_total), None
        ]
    )


def main():
    json_file = open('params.json')
    json_str = json_file.read()
    config = json.loads(json_str)


    ################
    # Train Loader #
    ################
    train_loader = __load_data(config)

    #######################
    # Loading the trained #
    ######## model ########
    model = NN(config['architecture'], is_maskable=True)


    dt = []
    for (root, dirs, files) in os.walk(config['model_path']):
        if 'SEARCH' in root or 'Lottery' in root:
            ckpt = [f for f in files if '.pt' in f]
    
            checkpoint = torch.load(root + '/' + ckpt[0], map_location=config['device'])
            model.load_state_dict(checkpoint)
            model = model.to(config['device'])
            model.eval()

            logging.info('Computing the train accuracy from:\n\t{}'.format(root + '/' + ckpt[0]))
            names = root.split('/')
            name = 'LOTTERY_TICKETS'
            if 'SEARCH' in root:
                name = names[1] + '__' + names[4]
            dt.append([name])

            ##############
            # Evaluating #
            ##############
            __make_evaluation(model, train_loader, config, dt)

    dt = pd.DataFrame(dt)
    name = config['model_path']
    dt.to_csv(f'{name}_train_acc.tsv', index=False, sep='\t')



if __name__ == "__main__":
    main()