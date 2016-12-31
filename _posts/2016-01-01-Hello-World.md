---
layout: post
title: Hello TensorFlow!
---
Recently I want to change my deeplearning tool from Matconvnet to Tensorflow. So I will write several blogs to record my learning step by step.

## MNIST

**Load MNIST Data**

```python
from tensorflow.examples.tutotials.mnist import input_data
mnist =  input_data.read_data_sets('MNIST_data',one_hot=True)

```
