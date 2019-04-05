# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 10:17:56 2019

@author: mnallanatha
"""

import tensorflow as tf
import numpy as np
def squash(s,axis=-1,epsilon=1e-7,name=None): #
    with tf.name_scope(name,default_name="squash"):
        squared_norm=tf.reduce_sum(tf.square(s),axis=axis,keepdims=True)
        safe_norm=tf.sqrt(squared_norm+epsilon) #  fix 0 division pproblem. # safe norm enable to avoid gradient vanishing
        squash_factor=squared_norm/(1.+squared_norm)
        unit_vector=s/safe_norm
        return squash_factor*unit_vector
    
def safe_norm(s,axis=-1,epsilon=1e-7,keepdims=False,name=None):
    with tf.name_scope(name,default_name="safe_norm"):
        squared_norm=tf.reduce_sum(tf.square(s),axis=axis,keepdims=keepdims)
        return tf.sqrt(squared_norm+epsilon)
# feed the input images
# Input Layer, grayscale image, size 28,28
X=tf.placeholder(shape=[None,28,28,1],dtype=tf.float32,name="X") # None <=> no batch size specified, signe channel

# primary capsule
# for each digit in will ouput 32 map, each containing 6x6 grid of 8 dimensionnal vector

caps1_n_maps=32
caps1_n_caps=caps1_n_maps*6*6 # 1152 primary capsules
caps1_n_dims=8

conv1_params={
        "filters" : 256,
        "kernel_size" : 9,
        "strides": 1,
        "padding": "valid",
        "activation" : tf.nn.relu,
        }
conv2_params={
        "filters" : caps1_n_maps*caps1_n_dims, # 256 conv filter,
        "kernel_size" : 9,
        "strides": 2,
        "padding": "valid",
        "activation" : tf.nn.relu,
        }

conv1=tf.layers.conv2d(X,name='conv1', **conv1_params)
conv2=tf.layers.conv2d(conv1,name='conv2',**conv2_params)
caps1_raw=tf.reshape(conv2,[-1,caps1_n_caps,caps1_n_dims],name="caps1_raw") # reshape to get 8D Vectors

caps1_output=squash(caps1_raw,name="caps1_output") # ensure lenght is between 0 and 1.

# digit capsules

# each capsules will compute an output based on the primary capsule output.
# the capsule layer contain 10 capsules, one for each digit.
caps2_n_caps=10
caps2_n_dims=16 # transform à 8D vector to 16

init_sigma=0.1
W_init=tf.random_normal( # variable initialize randomly
        shape=(1,caps1_n_caps,caps2_n_caps,caps2_n_dims,caps1_n_dims),
        stddev=init_sigma,dtype=tf.float32,name="W_init")
W=tf.Variable(W_init,name="W")

# tile the matrix for batchsize, 

batch_size=tf.shape(X)[0]
W_tiled=tf.tile(W,[batch_size,1,1,1,1],name="W_tiled")

#replication of u vector, to do multiplication

caps1_output_expanded=tf.expand_dims(caps1_output,-1,name="caps1_output_expanded") #  convert (batchsize,1152,8) to  (batchsize,1152,8,1) (column vector instead of array)
caps1_output_tile=tf.expand_dims(caps1_output_expanded,2,name="caps1_output_tile") # convert   (batchsize,1152,8,1)  to   (batchsize,1152,1,8,1) 
caps1_output_tiled=tf.tile(caps1_output_tile,[1,1,caps2_n_caps,1,1],name="caps_output_tiled")# convert (batchsize,1152,1,8,1)  to  (batchsize,1152,10,8,1) 
caps2_predicted=tf.matmul(W_tiled,caps1_output_tiled,name="caps2_predicted") #(W.u)

#  routing by agreement round1

# set all weights to zero
raw_weights=tf.zeros([batch_size,caps1_n_caps,caps2_n_caps,1,1],dtype=np.float32,name="raw_weights")

# round 1
routing_weights=tf.nn.softmax(raw_weights,axis=2,name="routing_weights")
weighted_predictions=tf.multiply(routing_weights,caps2_predicted,name="weighted_predictions")
weighted_sum=tf.reduce_sum(weighted_predictions,axis=1,keepdims=True,name="weighted_sum") #  s=somme(c.u)

caps2_output_round_1=squash(weighted_sum,axis=-2,name="caps2_output_round1")

# round 2 

caps2_predicted

caps2_output_round_1

# to match dimensions

caps2_output_round_1_tiled=tf.tile(
        caps2_output_round_1,[1,caps1_n_caps,1,1,1], # caps1_n_caps=1152
        name="caps2_output_round_1_tiled")

agreement=tf.matmul(caps2_predicted,caps2_output_round_1_tiled,transpose_a=True,name="agreemet")

raw_weights_round_2=tf.add(raw_weights,agreement,name="raw_weights_round_2")

routing_weights_round_2=tf.nn.softmax(raw_weights_round_2,axis=2,name="routing_weights_round_2")

weighted_predictions_round_2=tf.multiply(routing_weights_round_2,caps2_predicted,name="weighted_prediction_round_2")

weighted_sum_round_2=tf.reduce_sum(weighted_predictions_round_2,axis=1,keepdims=True,name="weighted_sum_round_2")

caps2_output_round_2=squash(weighted_sum_round_2,axis=-2,name="caps2_output_round_2")

caps2_output=caps2_output_round_2

y_proba = safe_norm(caps2_output,axis=-2,name="y_proba")

y_proba_argmax=tf.argmax(y_proba,axis=2,name="y_proba")

y_proba_argmax
#class predicted by caps network

y_pred=tf.squeeze(y_proba_argmax,axis=[1,2],name="y_pred")
#remove all 1 dimensions [2,1,1]=> [2]
y_pred