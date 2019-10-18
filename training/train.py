import numpy as np

import torch
import torch.nn as nn
from torchvision import datasets
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

import os
import sys
import json
import time
import logging

sys.path.append('../')
from pytorch.net import NN
from pytorch.utils import train, get_optimizer



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


def main():
    logging.basicConfig(level=logging.DEBUG,
                        format="[%(asctime)s %(filename)s] %(message)s")
    json_file = open('params.json')
    json_str = json_file.read()
    config = json.loads(json_str)

    # Create sub_working_dir
    sub_working_dir = '{}/{}/try{}/{}'.format(
        config['working_dir'],
        config['model']['name'], 
        config['try'],
        # time.strftime("%Y%m%d%H%M%S", time.localtime())
        '{}_{}/{}/{}_{}'.format(
            time.strftime("%Y", time.localtime()),
            time.strftime("%m", time.localtime()),
            time.strftime("%d", time.localtime()),
            time.strftime("%H", time.localtime()),
            time.strftime("%S", time.localtime())
        ) 
    )
    if not os.path.exists(sub_working_dir):
        os.makedirs(sub_working_dir)
    config["sub_working_dir"] = sub_working_dir
    logging.info("sub working dir: %s" % sub_working_dir)

    # Creat tf_summary writer
    config["tensorboard_writer"] = SummaryWriter(sub_working_dir)
    logging.info("Please using 'python -m tensorboard.main --logdir={}'".format(sub_working_dir))

    # loading the dataset
    train_loader, valid_loader, test_loader = __load_data(config)

    # Creating the model
    model = NN(config['model']['architecture'], is_maskable = True)
    model = model.to(config['device'])

    # Getting the criterion, optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(config, model)

    ###############
    # Begin Train #
    ###############
    train(model, train_loader, valid_loader, criterion, optimizer, config['train']['epochs'], config['train']['print_every'], config['device'])

    ##############
    # Begin Test #
    ##############
    # initialize lists to monitor test loss and accuracy
    test_loss = 0.0
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

    # calculate and logging.info avg test loss
    test_loss = test_loss/len(test_loader.sampler)
    logging.info('Test Loss: {:.6f}\n'.format(test_loss))

    for i in range(10):
        if class_total[i] > 0:
            logging.info('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                str(i), 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            logging.info('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

    logging.info('Test Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))    



if __name__ == "__main__":
    main()