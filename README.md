# Deep Compression for PyTorch Model Deployment on Microcontrollers

In this repository, you can find the source code of the paper *"Deep Compression for PyTorch Model Deployment on Microcontrollers"*.

This work follows the paper [Efficient Neural Network Deployment for Microcontroller](https://arxiv.org/abs/2007.01348) by Hasan Unlu. You can find the repository of the source code of that paper [here](https://github.com/hasanunlu/neural_network_deployment_for_uC).

## Dependencies
These are the only versions tested; therefore, other versions may be incompatible.
* Python 3.8.5
* PyTorch 1.8
* Tensorboard 2.4.1
* Neural Network Intelligence (NNI) 2.1

## Usage
Running the `generator.py` will generate `main.c` and `main.h` files in the outputs folder. `helper_functions.h` file is required by `main.c`.

Only two network architectures are included in this generator. To switch between the two, change this line in `generator.py`:
```python
dataset_name = 'mnist' # change this for different networks. can be 'mnist' or 'cifar10'
```

Some networks might be sensitive to input activation quantization. To disable input quantization, change this line in `generator.py`:
```python
quantize_input = dataset_name != 'cifar10' # change this for input quantization. can be True or False
```

You can also use this generator with networks other than **LeNet-5** or the **CIFAR-10 test network** implemented. You need to create the network using PyTorch building blocks and adjust the optimizer. Supported PyTorch building blocks are:
* Conv2d
* MaxPool2d
* Linear
* Flatten
* ReLU

You can look at our LeNet-5 implementation as a reference to supported model implementations:
```python
import torch.nn as nn

nn.Sequential (
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
```