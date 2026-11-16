# Convolutional Neural Network implemented in C++

A implementation of CNNs in C++ built on top of a custom Tensor library (inspired by my Linear Algebra class!). Single threaded CPU design (currently) and focusing on image recognition with rigorous test coverage.

## Features

* **Tensor Library:** Templated Tensor library for any rank tensor. Supports basic operations such as elementwise addition, multiplication, inner product, equality comparisons, tensor scaling and collapse, and a few activation functions (ReLU).
* **Supported Neural Network Layers:** Support for Convolutional, Activation, Min-Max Normalization, Flattening, and Dense Layer. Uses virtual Layer interface for forward pass and backward propagation. 
* **Neural Network Class:** Neural network class designed for simple addition of various layers using the Layer interface. Supports forward pass and back propagation.
* **Training System (in progress)** CPU single threaded training engine for labeled images.  
