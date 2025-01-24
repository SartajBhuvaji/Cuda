# CUDA-Optimized Convolutional Neural Network

## Overview
Designed and developed a Convolution Neural Network (CNN) library using CUDA C++, which allows for efficient training and inference of deep learning models on NVIDIA GPUs. The library features a modular design that allows for easy extension and customization of network architectures. It provides an implementation of the forward and backward passes for convolutional layers, dense layers, and pooling operations. The library also includes support for popular activation functions, loss functions, and optimization algorithms. I then train a simple CNN network on the CIFAR 10 dataset achieving 85% accuracy.

## Features
- **Activation Functions**: Implements ReLU, Leaky ReLU, Sigmoid, Tanh, ELU, SELU, and Softmax using CUDA.
- **Backpropagation**: Efficient gradient computation and optimization using CUDA.
- **Convolutional Operations**: Includes forward and backward passes for convolutional layers.
- **Dense Layers**: Supports matrix operations and dropout functionality for fully connected layers.
- **Image Handling**: Efficient loading and preprocessing of image data directly into GPU memory.
- **Loss Functions**: Implements cross-entropy loss with GPU acceleration.
- **Max Pooling**: Provides CUDA-accelerated max pooling operations.
- **Optimization Algorithms**: Features implementations of SGD with momentum and weight decay.
- **Data Preprocessing**: Handles normalization and one-hot encoding of labels.

## CUDA Version
- nvcc: NVIDIA (R) Cuda compiler driver
- Copyright (c) 2005-2024 NVIDIA Corporation
- Built on Fri_Jun_14_16:44:19_Pacific_Daylight_Time_2024
- Cuda compilation tools, release 12.6, V12.6.20
- Build cuda_12.6.r12.6/compiler.34431801_0

## Documentation
- [NVIDIA CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/index.html)
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)

