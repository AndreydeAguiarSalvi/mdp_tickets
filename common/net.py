# https://stackoverflow.com/questions/51748138/pytorch-how-to-set-requires-grad-false

from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F


class NN(nn.Module):

    def __init__(self, layers_sizes, is_maskable):
        ''' Builds a feedforward network with arbitrary hidden layers.
        
            Arguments
            ---------
            hidden_layers: list of integers, the sizes of all the layers
            is_maskable: boolean if is necessary to create masks and copies
        
        '''
        super().__init__()
        self.input_size = layers_sizes[0]
        self.output_size = layers_sizes[-1]
        self.is_maskable = is_maskable
        
        # Creating the Layers
        self.layers = nn.ModuleList([nn.Linear(layers_sizes[0], layers_sizes[1])])
        hidden_sizes = zip(layers_sizes[1:-2], layers_sizes[2:-1])
        self.layers.extend([nn.Linear(h1, h2) for h1, h2 in hidden_sizes])
        output = nn.Linear(layers_sizes[-2], layers_sizes[-1])
        self.layers.extend([output])

        
        if self.is_maskable:
            # Creating the masks and the copies
            self.masks = nn.ModuleList([])
            copies = []
            
            hidden_masks = zip(layers_sizes[:-1], layers_sizes[1:])
            self.masks.extend([nn.Linear(h1, h2) for h1, h2 in hidden_masks])
            with torch.no_grad():
                for mask in self.masks:
                    mask.weight.data.fill_(1.)
                    mask.bias.data.fill_(1.)
                    for param in mask.parameters():
                        param.requires_grad = False
            
            for fc in self.layers:
                with torch.no_grad():
                    copy = deepcopy(fc)
                    for param in copy.parameters():
                        param.requires_grad = False
                    copies.append(copy)
            self.copies = nn.ModuleList([*copies])
                
    
    def forward(self, x):
        x = x.view(-1, self.input_size)

        if self.is_maskable:
            for fc, mask in zip(self.layers, self.masks):
                with torch.no_grad():
                    fc.weight.data = fc.weight.data.mul(mask.weight.data)
        
        for i in range(len(self.layers) - 1):
            x = F.leaky_relu(self.layers[i](x))
        x = F.softmax(self.layers[-1](x))

        return x


    def reward(self):
        for fc, cp in zip(self.layers, self.copies):
            with torch.no_grad():
                fc.weight.data = deepcopy(cp.weight.data)
                fc.bias.data = deepcopy(cp.bias.data)


    def set_new_masks(self, masks):
        for new_mask, my_mask in zip(masks, self.masks):
            if new_mask.weight.shape != my_mask.weight.shape:
                raise RuntimeError('The mask received do net have the proper dimension')
            # elif new_mask.weight.is_cuda is not my_mask.weight.is_cuda:
            #     raise RuntimeError('The maks received is not in the same device')
        self.masks = masks


    def mask_to(self, device):
        for msk in self.masks:
            msk = msk.to(device)


    def get_layer(self, i):
        return self.layers[i]
    

    def get_mask(self, i):
        return self.masks[i]
    
    
    def get_copy(self, i):
        return self.copies[i]