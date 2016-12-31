---
layout: post
title: Hello TensorFlow!
---
Recently I want to change my deeplearning tool from Matconvnet to Tensorflow. 
So I will write several blogs to record my learning step by step.

## MNIST

### Perpare
```python
#Load Data
from tensorflow.examples.tutotials.mnist import input_data
mnist =  input_data.read_data_sets('MNIST_data',one_hot=True)
#Start Session
import tensorflow as tf
sess = tf.InteractiveSession()

#TensorFlow does its heavy operation outside Python. 
#The role of the Python code is therefore to build the external compuation graph. 

#Build a Softmax Regression Model
x = tf.placeholder(tf.float32,shape = [None,784]) 
y_ = tf.placeholder(tf.float32,shape = [None,10]) 
#placeholder is a value that filled by Tensorflow when running a computation.
#784 is the dim of a flattened 28 by 28 pixel MNIST image; 
#10 is the dim for 10 hand-crafted number.

#Variables
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
sess.run(tf.global_variables_initializer()) # re-initialized

y = tf.matmul(x,W)+b
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y,y_))
#reduce_mean takes the average over the loss.
```
###Train the Model
```python
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

```