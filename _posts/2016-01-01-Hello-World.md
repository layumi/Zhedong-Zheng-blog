---
layout: post
title: Hello TensorFlow!
---
Recently I want to change my deeplearning tool from Matconvnet to Tensorflow. 
So I will write several blogs to record my learning step by step.

## MNIST

```python
#Load Data
from tensorflow.examples.tutotials.mnist import input_data
mnist =  input_data.read_data_sets('MNIST_data',one_hot=True)
#Start Session
import tensorflow as tf
sess = tf.InteractiveSession()

```