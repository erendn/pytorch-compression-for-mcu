# Deep Compression for PyTorch Model Deployment on Microcontrollers

In this repository, you can find the source code of the paper *"Deep Compression for PyTorch Model Deployment on Microcontrollers"*.

This work follows the paper [Efficient Neural Network Deployment for Microcontroller](https://arxiv.org/abs/2007.01348). You can find the repository for the source code of that paper [here](https://github.com/hasanunlu/neural_network_deployment_for_uC).

# Dependencies
These are the only versions tested; therefore, other versions may be incompatible.
* Python 3.8.5
* Neural Network Intelligence (NNI) 2.1
* PyTorch 1.8
* Tensorboard 2.4.1

# Usage
Running the `generator.py` will generate `main.c` and `main.h` files in outputs folder. `helper_functions.h` file is required by main.c.

Only two network architectures are supported by this generator. To switch between the two, change this line in `generator.py`:
```python
dataset_name = 'mnist' # change this for different networks. can be 'mnist' or 'cifar10'
```