import os
import torch
import logging
import numpy as np
import pandas as pd
from pytorch.state import State
import torch.optim as optim


def validation(model, testloader, criterion, device='cuda:0'):
    accuracy = 0
    test_loss = 0
    model.eval()
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        equality = (labels.data == output.max(1)[1])
        # Accuracy is number of correct predictions divided by all predictions, just take the mean
        accuracy += equality.type_as(torch.FloatTensor()).mean()

    return test_loss/len(testloader), accuracy/len(testloader)


def train(model, trainloader, testloader, criterion, optimizer, epochs=5, print_every=40, device='cuda:0'):
    
    steps = 0
    running_loss = 0
    for e in range(epochs):
        # Model in training mode, dropout is on
        model.train()
        accuracy = 0
        for images, labels in trainloader:
            steps += 1
            
            # Flatten images into a 784 long vector
            # images.resize_(images.size()[0], 784)
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            equality = (labels.data == output.max(1)[1])
            accuracy += equality.type_as(torch.FloatTensor()).mean()

            if steps % print_every == 0:
                # Turn off gradients for validation, will speed up inference
                with torch.no_grad():
                    valid_loss, valid_acc = validation(model, testloader, criterion)
                
                logging.info(
                    'Epoch - Step: [{}/{} - {}] train_loss[{:.5f}] train_acc[{:.3f}] valid_loss[{:.5f}] valid_acc[{:.3f}]'.format(
                        e+1, epochs, steps,
                        # running_loss/print_every, accuracy/print_every, 
                        running_loss/len(trainloader), accuracy/len(trainloader), 
                        valid_loss, valid_acc
                    )
                )
                
                running_loss = 0
                
                # Make sure dropout and grads are on for training
                model.train()


def create_environment(masks, protocol):
    env = []
    
    if 'ROW' in protocol: 
        for layer in range(len(masks) - 1):
            for row in range(masks[layer].weight.shape[0]):
                my_dict = {}
                my_dict['layer{}_row{}'.format(layer, row)] = [layer, row]
                env.append(my_dict)
        
    if 'ELEMENT' in protocol:
        for layer in range(len(masks)):
            for row in range(masks[layer].weight.shape[0]):
                for col in range(masks[layer].shape[1]):
                    my_dict = {}
                    my_dict['layer{}_row{}_col{}'.format(layer, row, col)] = [layer, row, col] 
        
    return env


"""
    Returns the Q quality by a gived state and action
    If action is None, so the q_table will be initialized by that state.
"""
def q_value(q_table, state, action=None):
    
    if state not in q_table:
        q_table[state] = np.zeros(len(state.environment))
        
    if action is None:
        return q_table[state]
    
    return q_table[state][action]


def get_optimizer(config, net):
    optimizer = None

    params = net.parameters()

    # Initialize optimizer class
    if config['optimizer']['type'] == 'adam':
        logging.info('Using Adam optimizer')
        optimizer = optim.Adam(params, weight_decay=config['optimizer']['weight_decay'])
    elif config['optimizer']['type'] == 'amsgrad':
        logging.info('Using AMSGRAD optimizer')
        optimizer = optim.Adam(params, weight_decay=config['optimizer']['weight_decay'],
                               amsgrad=True)
    elif config['optimizer']['type'] == 'rmsprop':
        logging.info('Using RMSPROP optimizer')
        optimizer = optim.RMSprop(params, weight_decay=config['optimizer']['weight_decay'])
    elif config:
        logging.info("Using SGD optimizer.")
        optimizer = optim.SGD(params, momentum=config['optimizer']['momentum'],
                              weight_decay=config['optimizer']['weight_decay'],
                              nesterov=(config['optimizer']['type'] == 'nesterov'))
    else:
        raise RuntimeError('No one optimizer was found.')
    
    return optimizer


def save_checkpoint(state_dict, config):
    checkpoint_path = os.path.join(config['sub_working_dir'], 'model.pth')
    torch.save(state_dict, checkpoint_path)
    logging.info('Model checkpoint saved to %s' % checkpoint_path)


def q_table_loader(path):
    tsv = pd.read_csv(path, sep = '\t')
    q_table = dict()
    for i in range(tsv.shape[0]):
        string = tsv.iloc[i, 0]
        st = State(string)
        values = list(tsv.iloc[i, 1:])
        q_table[st] = values

    return q_table


def q_table_saver(q_table, path, name):
    result = []
    for key, value in q_table.items():
        line = []
        line.append(key)
        for x in value:
            line.append(x)
        result.append(line)
    final = pd.DataFrame(result)
    final.to_csv(path + name, sep='\t', index=False, header=False)


def policy_loader(path):
    tsv = pd.read_csv(path, sep = '\t')
    policy = []
    for i in range(0, len(tsv)):
        x = tsv.iloc[i][0].split(':')
        values = [int(s) for s in x[1].split('[')[1].split(']')[0].split(',')]
        keys = x[0].split('{\'')[1].split('\'')[0]
        policy.append({keys : values})
        
    return policy