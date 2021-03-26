import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torchvision.datasets.mnist import MNIST

import utils
import math
import copy
from pathlib import Path

from nni.compression.pytorch.compressor import PrunerModuleWrapper
from nni.compression.pytorch.compressor import QuantizerModuleWrapper
from nni.algorithms.compression.pytorch.quantization.quantizers import NaiveQuantizer

import torchvision.transforms as transforms

import torchvision.datasets as dset
import torchvision.transforms as T
import torch.nn.functional as F

import numpy as np

dataset_name = 'mnist' # change this for different networks. can be 'mnist' or 'cifar10'

quantize_input = dataset_name != 'cifar10' # change this for input quantization. can be True or False

pre_trained = True # change this if your model is pre-trained. can be True or False

use_cuda = False # change this if yor GPU supports CUDA. can be True or False

dtype = torch.float32

if use_cuda:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

if dataset_name == 'cifar10':
    # Example network on CIFAR10 dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = dset.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    data_train_loader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True)

    testset = dset.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    data_test_loader = torch.utils.data.DataLoader(testset, batch_size=512,
                                             shuffle=False)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    print_every = 50
    print('using device:', device)

    # Neural network architecture
    # Current code only supports conv2d-ReLU-maxPool2d pairs together and Linear
    model = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
        nn.ReLU(),

        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

        nn.Conv2d(32, 16, kernel_size=5, stride=1, padding=2),
        nn.ReLU(),

        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

        nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
        nn.ReLU(),

        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

        nn.Flatten(),

        nn.Linear(4*4*32, 10),
    )

elif dataset_name == 'mnist':
    # Example network(Lenet-5) on MNIST dataset
    data_train = MNIST('./data/mnist',
                       download=True,
                       transform=transforms.Compose([
                           transforms.Resize((32, 32)),
                           transforms.ToTensor()]))

    data_test = MNIST('./data/mnist',
                      train=False,
                      download=True,
                      transform=transforms.Compose([
                          transforms.Resize((32, 32)),
                          transforms.ToTensor()]))

    data_train_loader = DataLoader(data_train, batch_size=256, shuffle=True)
    data_test_loader = DataLoader(data_test, batch_size=1024)


    print_every = 50
    print('using device:', device)


    # Lenet-5 architecture
    model = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        nn.Flatten(),
        nn.Linear(5*5*16, 120),
        nn.ReLU(),
        nn.Linear(120, 84),
        nn.ReLU(),
        nn.Linear(84, 10),
    )

Path("./saves/"+dataset_name).mkdir(parents=True, exist_ok=True)
Path("./outputs/"+dataset_name).mkdir(parents=True, exist_ok=True)

best_model = None
initial_acc = None

if not pre_trained:
    best_model, initial_acc = utils.train_model(model, data_train_loader, data_test_loader, 4, optim.Adam(model.parameters(), lr=2e-3), True)
    print(best_model)
    torch.save(best_model, './saves/'+dataset_name+'/original.pt')

# +-+-+-+ BINARY SEARCH FOR OPTIMAL SPARSITY VALUE +-+-+-+
current_model = torch.load("./saves/"+dataset_name+"/original.pt").to(device) # load network. make sure to name correctly for pre-trained networks
best_model = current_model

if initial_acc is None:
    initial_acc = utils.evaluate_model(best_model, data_test_loader)

tolerated_acc_loss = 0.01 # manual parameter to tolerate accuracy loss
min_search_step = 0.001 # manual parameter to stop the binary search
step = 0.5
sparsity = 0.5
best_sparsity = 0
while step > min_search_step: # continue until min search step is crossed
    model = copy.deepcopy(current_model)
    utils.level_prune_model(model, [{ 'sparsity': sparsity, 'op_types': ['default'] }])
    _, acc = utils.train_model(model, data_train_loader, data_test_loader, 4, optimizer=optim.Adam(model.parameters(), lr=2e-3))
    step /= 2
    if acc >= initial_acc - tolerated_acc_loss:
        best_model = model
        best_sparsity = sparsity
        sparsity += step
        print("Current sparsity: " + str(best_sparsity))
    else:
        sparsity -= step
final_acc = utils.evaluate_model(best_model, data_test_loader)
result_str = "Initial Accuracy: " + str(initial_acc) + " Sparsity: " + str(best_sparsity) + " Accuracy: " + str(final_acc)
print("Best sparsity found! " + result_str)

torch.save(best_model, './saves/'+dataset_name+'/pruned.pt')

middle_acc = utils.evaluate_model(best_model, data_test_loader)

middle_model = torch.load("./saves/"+dataset_name+"/original.pt").to(device) # load network from original.py
model_params = list(best_model.parameters())
q_count = 0
with torch.no_grad():
    for param in middle_model.parameters():
        flat_weights = param.flatten().numpy()
        for idx in range(len(flat_weights)):
            if len(param.shape) == 1:
                param[idx].data += torch.from_numpy(np.array(model_params[q_count][idx])).data - param[idx].data
            elif len(param.shape) == 2:
                i_0, i_1 = divmod(idx, param.shape[1])
                param[i_0][i_1].data += torch.from_numpy(np.array(model_params[q_count][i_0][i_1])).data - param[i_0][i_1].data
            elif len(param.shape) == 4:
                i_2, i_3 = divmod(idx, param.shape[3])
                i_1, i_2 = divmod(i_2, param.shape[2])
                i_0, i_1 = divmod(i_1, param.shape[1])
                param[i_0][i_1][i_2][i_3].data += torch.from_numpy(np.array(model_params[q_count][i_0][i_1][i_2][i_3])).data - param[i_0][i_1][i_2][i_3].data
        q_count += 1

# +-+-+-+ 8-BIT WEIGHT QUANTIZATION +-+-+-+
quantized_model = middle_model
print("Accuracy after pruning: "+str(middle_acc))
config_list = [{ 
    'quant_types': ['weight'],
    'quant_bits': 8, 
    'op_types': ['Conv2d', 'Linear']
}]
quantizer = NaiveQuantizer(quantized_model, config_list)
quantizer.compress()
final_acc = utils.evaluate_model(quantized_model, data_test_loader)
print("Accuracy after weight quantization (8-bits): "+str(final_acc))

# +-+-+-+ 8-BIT ACTIVATION QUANTIZATION +-+-+-+
activation_bounds = utils.get_activation_bounds(quantized_model, data_train_loader)
print(activation_bounds)
final_acc = utils.evaluate_quantized_model(quantized_model, data_test_loader, activation_bounds, quantize_input = quantize_input)
print("Accuracy after activation quantization (8-bits): "+str(final_acc))

best_model = quantized_model
layer_parameters = {}
# {
#   'layer_name': {
#                   'scale': ...,
#                   'bias': [ ... ],
#                   'quantized: [ ... ],
#                   'indices': [ ... ]
#                 },
# }
for name, param in best_model.state_dict().items(): # extract weights and biases from the model
    name = name.replace('.module', '')
    if 'mask' in name or 'old' in name:
        continue
    arr = param.cpu().numpy()
    shape_of_params = arr.shape
    print(shape_of_params)
    param_size =  len(arr.flatten())
    print(name)
    layer_name = 'w_'+name.split('.', 1)[0]
    if layer_name not in layer_parameters.keys():
        layer_parameters[layer_name] = {}
    layer_parameters[layer_name]['scale'] = quantizer.layer_scale[layer_name.split('_')[1]].numpy()
    if "bias" in name:
        layer_parameters[layer_name]['bias'] = arr.flatten()
    elif "weight" in name:
        quantized, indices = utils.sparse_matrix_1d(arr.flatten())
        layer_parameters[layer_name]['quantized'] = torch.div(torch.Tensor(np.array(quantized)), torch.Tensor(layer_parameters[layer_name]['scale'])).type(torch.int8).numpy()
        layer_parameters[layer_name]['indices'] = indices

torch.save(best_model, './saves/'+dataset_name+'/quantized.pt')

# weights header and network generator. This generates main.h and main.c
weights_file = open('./outputs/'+dataset_name+'/main.h', 'w')
weights_file.write('typedef float data_t;\n\
typedef int8_t quan_t;\n\
typedef uint8_t index_t;\n\n')

c_file = open('./outputs/'+dataset_name+'/main.c', 'w')
c_file.write('// Initial Accuracy: '+str(initial_acc)+' Final Accuracy: '+str(final_acc)+'\n\
#include <stdio.h>\n\
#include <string.h>\n\
#include <stdint.h>\n\
#include <stdlib.h>\n\
#include "main.h"\n\
#include "helper_functions.h"\n')

c_file.write('\n\
int main()\n\
{\n')

test_vector_batch = 5
test_vector_index_in_batch = 6

test_vector = None

for i, (images, labels) in enumerate(data_test_loader): # sample input data
    if i == test_vector_batch:
        test_vector_index_in_batch = 3
        img = images[test_vector_index_in_batch].numpy()        
        test_vector = images
        
        weights_file.write('const '+('quan_t' if quantize_input else 'data_t')+' test['+str(img.size)+']={')
        data = images[test_vector_index_in_batch].flatten()
        for x in range(len(data)):
            if x != 0:
                weights_file.write(',')
            if quantize_input:
                weights_file.write(str(int(data[x].item() * 255 - 128)))
            else:
                weights_file.write(str(data[x].item() if data[x] != 0 else 0))
        weights_file.write('};\n')
        break

result = test_vector
input_size = result.shape[1]*result.shape[2]*result.shape[3]
L = np.empty(0)
L = np.append(L, np.uint32(input_size))

previous_padding = None

index = 0

meta_list = list()

c_file.write('\t twoD_t meta_data'+str(index)+' = {\n\
                           .r = '+ str(result.shape[2]) +',\n\
                           .c = '+str(result.shape[3])+',\n\
                           .channel = '+str(result.shape[1])+',\n\
                           .scale = '+str(1/255)+',\n\
                           .zero_quan = -128,\n\
                           .data = buffer'+str(index%2)+',\n\
                           .indices = NULL,\n\
                           .bias = NULL\n\
                       };\n\n')

meta_list.append(('meta_data'+str(index),(0,0,0)))
prev_channel_size = 1
max_kernel_size = 0

for i in best_model:
    result = i(result)
    if isinstance(i, nn.Conv2d):
        previous_padding = i.padding[0]
    if 'ReLU' in str(i):
        continue
    if isinstance(i, nn.MaxPool2d):
        index += 1
        size = prev_channel_size * result.shape[2] * result.shape[3]
        if max_kernel_size < size:
            max_kernel_size = size
        L = np.append(L, result.shape[1]*result.shape[2]*result.shape[3])
        c_file.write('\t twoD_t meta_data'+str(index)+' = {\n\
                           .r = '+str(result.shape[2])+',\n\
                           .c = '+str(result.shape[3])+',\n\
                           .channel = '+str(result.shape[1])+',\n\
                           .scale = '+str(activation_bounds[index - 1]['scale'])+',\n\
                           .zero_quan = '+str(activation_bounds[index - 1]['zero'])+',\n\
                           .data = buffer'+str(index%2)+',\n\
                           .indices = NULL,\n\
                           .bias = NULL\n\
                       };\n')
        meta_list.append(('meta_data'+str(index),(i.stride, i.kernel_size, previous_padding if previous_padding is not None else 0)))
        prev_channel_size = result.shape[1]
    
    if 'Linear' in str(i):
        index += 1
        L = np.append(L, result.shape[1])
        c_file.write('\t twoD_t meta_data'+str(index)+' = {\n\
                           .r = '+str(result.shape[1]) +',\n\
                           .c = 1,\n\
                           .channel = 1,\n\
                           .scale = '+str(activation_bounds[index - 1]['scale'])+',\n\
                           .zero_quan = '+str(activation_bounds[index - 1]['zero'])+',\n\
                           .data = buffer'+str(index%2)+',\n\
                           .indices = NULL,\n\
                           .bias = NULL\n\
                       };\n\n')
        meta_list.append(('meta_data'+str(index),(0,0,0)))

c_file.write('\n\t memcpy(buffer0, test, sizeof(test));\n')

c_file.write('\n\t printf("---Network starts---\\n");\n')


inx = np.argsort(L)

weights_file.write('\n\
quan_t buffer'+str(inx[-1]%2)+'['+str(int(L[inx[-1]]))+'];\n\
quan_t buffer'+str((inx[-1]+1)%2)+'['+str(int(L[inx[-2]]))+'];\n\
quan_t w_kernel['+str(max_kernel_size)+'];\n\
\n\
typedef struct twoD\n\
{\n\
    uint32_t r;\n\
	uint32_t c;\n\
	uint32_t in_channel;\n\
	uint32_t channel;\n\
    data_t scale;\n\
    quan_t zero_quan;\n\
    quan_t *data;\n\
    index_t *indices;\n\
	data_t *bias;\n\
} twoD_t;\n\n')

prev_shapes = None
prev_arr_name = None
is_bias_first = None

index = 0
    
for name, param in best_model.state_dict().items():
    name = name.replace('.module', '')
    if 'mask' in name or 'old' in name:
        continue
    arr = param.cpu().numpy()
    shape_of_params = arr.shape
    print(shape_of_params)
    param_size =  len(arr.flatten())
    print(name)
    layer_name = 'w_'+name.split('.', 1)[0]
    if "bias" in name:
        if is_bias_first is None:
            is_bias_first = True
        underscore_arr_name = 'w_'+name.replace('.', '_')
        array_name  = underscore_arr_name + '[' + str(param_size) + ']=' 
        weights_file.write('const data_t '+array_name+'{')
        for x in range(len(layer_parameters[layer_name]['bias'])):
            if x != 0:
                weights_file.write(',')
            weights_file.write(str(layer_parameters[layer_name]['bias'][x] if layer_parameters[layer_name]['bias'][x] != 0 else 0))
        weights_file.write('};\n')
    elif "weight" in name:
        if is_bias_first is None:
            is_bias_first = False
        # Saving 8-bit quantized weights
        underscore_arr_name = 'w_'+name.replace('.', '_')+'_quantized'
        array_name  = underscore_arr_name + '[' + str(len(layer_parameters[layer_name]['quantized'])) + ']='
        weights_file.write('const quan_t '+array_name+'{')
        for x in range(len(layer_parameters[layer_name]['quantized'])):
            if x != 0:
                weights_file.write(',')
            weights_file.write(str(layer_parameters[layer_name]['quantized'][x]))
        weights_file.write('};\n')
        # Saving indices of weight values
        underscore_arr_name = 'w_'+name.replace('.', '_')+'_indices'
        array_name  = underscore_arr_name + '[' + str(len(layer_parameters[layer_name]['indices'])) + ']='
        weights_file.write('const index_t '+array_name+'{')
        for x in range(len(layer_parameters[layer_name]['indices'])):
            if x != 0:
                weights_file.write(',')
            weights_file.write(str(layer_parameters[layer_name]['indices'][x]))
        weights_file.write('};\n')
        underscore_arr_name = 'w_'+name.replace('.', '_')
    else:
        continue

    if not is_bias_first and len(shape_of_params) == 1:
        index += 1
        if len(prev_shapes) > 2:
            print('Conv layer')
            out_channel = prev_shapes[0]
            in_channel = prev_shapes[1]
            c_file.write('\t conv2D'+\
            '(&'+meta_list[index-1][0]+', &'+prev_arr_name+'_2d, &'+meta_list[index][0]+', &reLU, '+str(meta_list[index][1][0])+', '+\
            str(meta_list[index][1][1])+', '+str(meta_list[index][1][1])+', '+str(meta_list[index][1][2])+');\n')
            
        else:
            print('linear layer')
            out_channel = 1
            in_channel = 1
            if len(meta_list) == (index+1):
                c_file.write('\t dot'+'(&'+meta_list[index-1][0]+', &'+prev_arr_name+'_2d, &'+meta_list[index][0]+', NULL);\n')
            else:
                c_file.write('\t dot'+'(&'+meta_list[index-1][0]+', &'+prev_arr_name+'_2d, &'+meta_list[index][0]+', &reLU);\n')
                
        weights_file.write(('const twoD_t '+prev_arr_name+'_2d = {\n'+\
                            '\t.r = '+ str(prev_shapes[-1]) +',\n'+\
                            '\t.c = '+str(prev_shapes[-2])+',\n'+\
                            '\t.in_channel = '+str(in_channel)+',\n'+\
                            '\t.channel = '+str(out_channel)+',\n'+\
                            '\t.scale = '+str(layer_parameters[layer_name]['scale'])+',\n\t.data = '+prev_arr_name+'_quantized,\n\t.indices = '+prev_arr_name+'_indices,\n'+\
                            '\t.bias = '+prev_arr_name+'\n'+\
                            '};\n\n'))
    
    elif is_bias_first and len(shape_of_params) != 1:
        index += 1
        if len(shape_of_params) > 2:
            print('Conv layer')
            out_channel = shape_of_params[0]
            in_channel = shape_of_params[1]
            c_file.write('\t conv2D'\
            +'(&'+meta_list[index-1][0]+', &'+underscore_arr_name+'_2d, &'+meta_list[index][0]+', &reLU, '+str(meta_list[index][1][0])+', '\
            +str(meta_list[index][1][1])+', '+str(meta_list[index][1][1])+', '+str(meta_list[index][1][2])+');\n')
        else:
            print('linear layer')
            out_channel = 1
            in_channel = 1
            if len(meta_list) == (index+1):
                c_file.write('\t dot(&'+meta_list[index-1][0]+', &'+underscore_arr_name+'_2d, &'+meta_list[index][0]+', NULL);\n')
            else:
                c_file.write('\t dot(&'+meta_list[index-1][0]+', &'+underscore_arr_name+'_2d, &'+meta_list[index][0]+', &reLU);\n')
                
        weights_file.write(('const twoD_t '+underscore_arr_name+'_2d = {\n'+\
                            '\t.r = '+ str(shape_of_params[-1]) +',\n'+\
                            '\t.c = '+str(shape_of_params[-2])+',\n'+\
                            '\t.in_channel = '+str(in_channel)+',\n'+\
                            '\t.channel = '+str(out_channel)+',\n'+\
                            '\t.scale = '+str(layer_parameters[layer_name]['scale'])+',\n\t.data = '+underscore_arr_name+'_quantized,\n\t.indices = '+underscore_arr_name+'_indices,\n'+\
                            '\t.bias = '+prev_arr_name+'\n'+\
                            '};\n\n'))

    prev_shapes = shape_of_params
    prev_arr_name = underscore_arr_name

c_file.write('\n\t print_twoD(&'+meta_list[-1][0]+', 0);\n')
c_file.write('\t printf("PREDICTION: %d\\n", get_class(&'+meta_list[-1][0]+'));\n')

class network_partial(nn.Module):
    def __init__(self, original_model):
        super(network_partial, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-6])
        
    def forward(self, x):
        x = self.features(x)
        return x

intermediate_network = network_partial(best_model)

weights_file.close()

c_file.write('\t return 0;\n}')

c_file.close()