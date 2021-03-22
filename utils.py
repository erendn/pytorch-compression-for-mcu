import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torch.nn.functional as F
import torchvision.datasets as dset

from nni.algorithms.compression.pytorch.pruning import LevelPruner
from nni.compression.pytorch.compressor import QuantizerModuleWrapper
from nni.compression.pytorch.compressor import PrunerModuleWrapper

import numpy as np
import time
import sys

if 0:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def evaluate_model(model, test_loader, process_time=False, print_out=False):
    """
    Find the accuracy of a model on the a dataset.

    Inputs:
    - model: A PyTorch Module giving the model to find the accuracy of
    - test_loader: A data loader object to receive test data
    - process_time: (Optional) Boolean to print the evaluation time
    - print_out: (Optional) Boolean to print the accuracy in each iteration

    Returns: The accuracy of the model
    """
    num_correct = 0
    num_samples = 0
    model.eval() # set model to evaluation mode
    with torch.no_grad():
        if process_time and print_out:
            start = time.process_time()
        for x, y in test_loader:
            x = x.to(device=device) # move to device, e.g. GPU
            y = y.to(device=device) # move to device, e.g. GPU
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        if process_time:
            end = time.process_time()
        acc = float(num_correct) / num_samples
        if print_out:
            print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
        if process_time and print_out:
            print("Time elapsed: " + str(end - start))
    return acc

def train_model(model, train_loader, validate_loader, epochs, optimizer, print_out=False):
    """
    Train a model on the given dataset using the PyTorch Module API.

    Inputs:
    - model: A PyTorch Module giving the model to train
    - train_loader: A data loader object to receive train data
    - validate_loader: A data loader object to receive validate data
    - epochs: (Optional) A Python integer giving the number of epochs to train for
    - optimizer: An Optimizer object we will use to train the model
    - print_out: (Optional) Boolean to print the accuracy in each iteration

    Returns: Best model and its accuracy
    """
    best_acc = -1
    best_model = None
    model = model.to(device=device)
    for e in range(epochs):
        for t, (x, y) in enumerate(train_loader):
            model.train()  # put model to training mode
            x = x.to(device=device)  # move to device, e.g. GPU
            y = y.to(device=device)

            scores = model(x)
            loss = F.cross_entropy(scores, y)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

            if t % 50 == 0:
                if print_out:
                    print('Iteration %d, loss = %.4f' % (t, loss.item()))
                acc = evaluate_model(model, validate_loader)
                if acc > best_acc:
                    best_acc = acc
                    best_model = model
                if print_out:
                    print()
    if print_out:
        print('Best accuracy found', best_acc)
    return best_model, best_acc

def get_activation_bounds(model, train_loader):
    """
    Find the activation bounds of a network on a training data.
    Activation bound is the minimum and maximum values a node
    in the network can have.

    Inputs:
    - model: A PyTorch Module giving the model to train
    - train_loader: A data loader object to receive train data

    Returns: Activation bounds, scale values, and zero indices for each layer 
    """
    bounds = {}
    # {
    #   'output_order': {
    #                   'min': ...,
    #                   'max': ...,
    #                   'scale: ...,
    #                   'zero': ...
    #                 },
    # }
    model.eval()
    with torch.no_grad():
        children = list(model.children()) # get all layers
        for x, y in train_loader:
            x = x.to(device=device)
            y = y.to(device=device)
            output = x
            order = 0
            for i in range(len(children)):
                output = children[i](output)
                # find the max and min activations for each layer
                if not isinstance(children[i], nn.Flatten) and (i + 1 == len(children) or isinstance(children[i + 1], QuantizerModuleWrapper) or isinstance(children[i + 1], nn.Conv2d) or isinstance(children[i + 1], nn.Linear) or isinstance(children[i + 1], nn.Flatten)):
                    if order not in bounds.keys():
                        bounds[order] = { 'min': sys.float_info.max, 'max': sys.float_info.min, 'scale': 0, 'zero': 0 }
                    if torch.max(output).item() > bounds[order]['max']:
                        bounds[order]['max'] = torch.max(output).item()
                    if torch.min(output).item() < bounds[order]['min']:
                        bounds[order]['min'] = torch.min(output).item()
                    order += 1
    for idx in bounds: # find the scale and the index of zero for each layer 
        bounds[idx]['scale'] = float((bounds[idx]['max'] - bounds[idx]['min']) / 255)
        bounds[idx]['zero'] = int(-(bounds[idx]['min'] / bounds[idx]['scale']) - 128)
    return bounds

def evaluate_quantized_model(model, test_loader, activation_bounds, quantize_input=True):
    """
    Find the accuracy of a quantized model on a test dataset. Contrary to the
    evaluate_model function, weights and activations are quantized to evaluate
    the model accurately.

    Inputs:
    - model: A PyTorch Module giving the model to find the accuracy of
    - test_loader: A data loader object to receive test data
    - activation_bounds: Activation bounds of the given model
    - quantize_input: (Optional) Boolean to quantize input activations

    Returns: The accuracy of the model
    """
    num_correct = 0
    num_samples = 0
    model.eval() # set model to evaluation mode
    with torch.no_grad():
        children = list(model.children()) # get all layers
        for x, y in test_loader:
            x = x.to(device=device) # move to device, e.g. GPU
            y = y.to(device=device) # move to device, e.g. GPU
            scores = x
            if quantize_input: # quantize input activations
                scores = torch.clamp(torch.add(torch.mul(x, 255), -128), -128, 127)
                scores = torch.floor(scores)
                scores = torch.div(torch.sub(scores, -128), 255)
            order = 0
            for i in range(len(children)):
                scores = children[i](scores)
                # quantize output activations of the layer
                if not isinstance(children[i], nn.Flatten) and (i + 1 == len(children) or isinstance(children[i + 1], QuantizerModuleWrapper) or isinstance(children[i + 1], nn.Conv2d) or isinstance(children[i + 1], nn.Linear) or isinstance(children[i + 1], nn.Flatten)):
                    scores = torch.clamp(torch.add(torch.div(scores, activation_bounds[order]['scale']), activation_bounds[order]['zero']), -128, 127)
                    scores = torch.floor(scores)
                    scores = torch.mul(torch.sub(scores, activation_bounds[order]['zero']), activation_bounds[order]['scale'])
                    order += 1
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
    return acc

def level_prune_model(model, config_list):
    """
    Prune a model with the given configuration using the Neural Network Intelligence's (NNI)
    LevelPruner tool.

    Inputs:
    - model: A PyTorch Module giving the model to prune
    - config_list: A configuration object for the LevelPruner

    Returns: Nothing
    """
    pruner = LevelPruner(model, config_list)
    pruner.compress()

def sparse_matrix_1d(array, max_bit_size=8):
    """
    Converts an array to compressed sparse column (CSC) format. 

    Inputs:
    - model: An array to be converted to CSC
    - max_bit_size: (Optional) An integer as the maximum bit width

    Returns: Nothing
    """
    values = []
    indices = []
    index_diff = 0
    for x in array:
        if x != 0:
            values.append(x)
            indices.append(index_diff)
            index_diff = 0
        if index_diff == 2 ** max_bit_size - 1:
            values.append(0)
            indices.append(index_diff)
            index_diff = 0
        index_diff += 1
    return values, indices