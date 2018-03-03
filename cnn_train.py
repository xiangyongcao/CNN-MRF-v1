# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 16:21:13 2017
@author: Xiangyong Cao
This code is modified based on https://github.com/KGPML/Hyperspectral
"""

from __future__ import print_function
import tensorflow as tf
import HSI_Data_Preparation
from HSI_Data_Preparation import num_train,Band, All_data,  Height, Width, Label, TrainIndex, TestIndex
from utils import patch_size, Post_Processing, convertToOneHot, draw
import numpy as np
import os
import scipy.io
from cnn_model import conv_net
import time

os.environ["CUDA_VISIBLE_DEVICES"]='1'   
DATA_PATH = os.getcwd()

def compute_prob_map():
    # Obtain the probabilistic map
    num_all = len(All_data['patch'])
    times = 20
    num_each_time = int(num_all / times)
    res_num = num_all - times * num_each_time
    Num_Each_File = num_each_time * np.ones((1,times),dtype=int)
    Num_Each_File = Num_Each_File[0]
    Num_Each_File[times-1] = Num_Each_File[times-1] + res_num
    start = 0
    prob_map = np.zeros((1,n_classes))
    for i in range(times):
        feed_x = np.reshape(np.asarray(All_data['patch'][start:start+Num_Each_File[i]]),(-1,n_input))
        temp = sess.run(softmax_output, feed_dict={x: feed_x})
        prob_map = np.concatenate((prob_map,temp),axis=0)
        start += Num_Each_File[i]
    
    prob_map = np.delete(prob_map,(0),axis=0)
    return np.array(prob_map)

start_time = time.time()
# Import HSI data
Training_data, Test_data = HSI_Data_Preparation.Prepare_data()
n_input = Band * patch_size * patch_size 

Training_data['train_patch'] = np.transpose(Training_data['train_patch'],(0,2,3,1))
Test_data['test_patch'] = np.transpose(Test_data['test_patch'],(0,2,3,1))
All_data['patch'] = np.transpose(All_data['patch'],(0,2,3,1))

Training_data['train_patch'] = np.reshape(Training_data['train_patch'],(-1,n_input))
Test_data['test_patch'] = np.reshape(Test_data['test_patch'],(-1,n_input))
All_data['patch'] = np.reshape(All_data['patch'],(-1,n_input))

# Parameters
learning_rate = 0.0001
Epoch = 500
batch_size = 200
display_step = 500
n_classes = 16

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# Construct model
pred = conv_net(x)
softmax_output= tf.nn.softmax(pred)

# Define loss and optimizer
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Define accuracy
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

predict_test_label = tf.argmax(pred, 1)

# Initializing the variables
init = tf.global_variables_initializer()


x_test, y_test = Test_data['test_patch'], Test_data['test_labels']
y_test_scalar = np.argmax(y_test,1) + 1
x_train, y_train = Training_data['train_patch'],Training_data['train_labels']

print("--------------------------------------------------------")
print("Begin training")
# Launch the graph
with tf.Session() as sess:
    sess.run(init)  
    # Training cycle
    for epoch in range(Epoch):
        training_iters = num_train / batch_size
        for iteration in range(training_iters):
            idx = np.random.choice(num_train, size=batch_size, replace=False)
            # Use the random index to select random images and labels.
            batch_x = Training_data['train_patch'][idx, :]
            batch_y = Training_data['train_labels'][idx, :]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, batch_cost, train_acc = sess.run([optimizer, cost, accuracy], 
                                        feed_dict={x: batch_x,y: batch_y})
        if epoch % 10 == 0: 
            print("Epoch", '%04d,' % (epoch), "Batch cost=%.4f," % (batch_cost),\
                      "Training Accuracy=%.4f" % (train_acc))    
    print("Training Finished!")
    print("--------------------------------------------------------")
    
    # MRF stage
    print("Post-processing...")
    prob_map = compute_prob_map()
    Seg_Label, seg_Label, seg_accuracy = Post_Processing(prob_map,Height,Width,n_classes,y_test_scalar,TestIndex)
    
    # Test model
    print("The Final Test Accuracy is :", sess.run(accuracy,feed_dict={x: x_test,y: y_test}))
    print("The Final Seg Accuracy is :", seg_accuracy) 
    
    print("--------------------------------------------------------")    
    end_time = time.time()
    print('The elapsed time is %.2f' % (end_time-start_time))    
    print("--------------------------------------------------------")
    