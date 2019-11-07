import os
import sys
import json
import torch
import logging
import numpy as np
sys.path.append('../')
from common.net import NN
from torchvision import datasets
import torchvision.transforms as transforms


def __load_data(config):
    # number of subprocesses to use for data loading
    num_workers = 0
    # how many samples per batch to load
    batch_size = config['batch_size']

    # convert data to torch.FloatTensor
    transform = transforms.ToTensor()

    test_data = datasets.MNIST(root='../data', train=False,
                                    download=False, transform=transform)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
        num_workers=num_workers)

    return test_loader


logging.basicConfig(level=logging.DEBUG,
                        format='[%(asctime)s %(filename)s] %(message)s')

def main():
    json_file = open('params.json')
    json_str = json_file.read()
    config = json.loads(json_str)

    ################
    # Train Loader #
    ################
    test_loader = __load_data(config)

    #######################
    # Loading the trained #
    ######## model ########
    model = NN(config['architecture'], is_maskable=True)
    for (_, _, files) in os.walk(config['model_path']):
        file = [s for s in files if 'pt' in s]
    
    checkpoint = torch.load(config['model_path'] + '/' + file[0], map_location=config['device'])
    model.load_state_dict(checkpoint)
    model = model.to(config['device'])
    model.eval()

    ##############
    # Evaluating #
    ##############
    classes = range(0, 10)
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    for images, labels in test_loader:
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