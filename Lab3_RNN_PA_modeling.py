#!/usr/bin/env python
# coding: utf-8

# # RF Power Amplifier Envelope Distortion Modeling
# 
# Modeling the behavior of an RF power amplifier is not a simple task. 
# In particular, modern RF power amplifiers exhibit significant nonlinear distortions that must be modeled in order to predict or correct for these distortions. 
# However, not only the current, but also all previous inputs to a power amplifier device must be considered in order to predict the correct power amplifier outputs. 
# In this example, we will explore different RNN structures to achieve this goal.
# 
# Note, this is a bit of a simplified example. Normally you would be operating complex-valued base-band 
# signals but for the sake of simplicity we focus on just modeling the power envelope of the signal.
# A bit of more background here: the RF Amplifier Operates an upconverted high-frequency signal. 
# However, the majority of the distortion "just" affect the "envelope" of the RF signal, thus, for modeling it we can begin with just the magnitude of the converted base-band signals. 

# We start, as always, by importing the necessary libraries. 

# In[1]:


import numpy as np
import tensorflow as tf
import tensorflow.keras as K
import matplotlib.pyplot as plt
from scipy.io import loadmat


# In[2]:


# this is how you import data from Matlab. 
# loadmat returns a dictionary
mat = loadmat("./Lab3_Data.mat") # assuming that the data is in the same folder...

x = mat['pa_input']
y = mat['pa_output']


# The next few lines section plot the behavior that should be modeled.
# The samples in x and y are a time-series, so their order corresponds to the time dimension. 
# Observe the scattering/hysteresis around the instantaneous behaviour. This is the dynamic/memory effect of the power amplifier.

# In[3]:


plt.figure(figsize=(10,3))
plt.subplot(1, 2, 1)
plt.scatter(x, y)
plt.xlabel("PA input"), plt.ylabel("PA output"), plt.grid()
plt.subplot(1, 2, 2)
plt.plot(x[50:200], label="input")
plt.plot(y[50:200], label="output")
plt.xlabel("Time index"), plt.ylabel("Magnitude"), plt.grid(), plt.legend()
plt.show()


# Let's use the first 60k samples in the dataset for training and the last 20k samples as validation data.
# Don't forget to scale/standardize your data.
# Additionally you will need to reshape the data, as explained in the next block. 

# In[4]:


# TODO:


# Let's define an RNN sequence-to-sequence model with a GRU layer with `6 units`. (GRU=gated recurrent unit).
# 
# A seq-to-seq model means, that with every time step, the RNN output is returned. \
# For that to happen, we need to specify `return_sequences=True` for the RNN layer.
# 
# The model should then look like:
# Input layer -> GRU layer (6) -> hidden/fully-connected layer-> output layer
# 
# To model the PA amplitude, our model in principle takes one input feature and one output feature.\
# However, training of RNNs requires to unroll the RNN structures in time, to be able to find proper gradients. \
# The depth of the unrolling is often referred as "timesteps", I suggest trying these in the range between 10 and 50. \
# While TF does the unrolling of the RNN automatically, we need to include the number of time-steps for unrolling in the first dimension of the shape argument of the input layer.
# 
# We also need to prepare the input and output training data accodingly, with data dims specified as [batch-dim, timesteps, features]. \
# Thus, in the second dimension (timesteps), consecutive samples should be correctly following one another. 
# 
# Try to observe the performance with an additional metric 'log MSE' and check the convergence using the logarithic metric.

# In[5]:


# TODO:

# input_layer = K.Input

# ...

# output_layer = 

# RNN_model = K.Model(input_layer, output_layer)


# In[6]:


# TODO:
#RNN_model.compile( )
#hist = RNN_model.fit(x_train, y_train, epochs=, batch_size=)


# Use the model's predict method to get an output. The validation datay you provide to the predict method has to follow the same arrangement as the training data.\
# Although RNNs benefit from longer training, you can reach a good converged performance with just 100 epochs. 
# 
# Plot the output to input amplitude for the power amplifier and the model and compare.

# In[7]:


# TODO: plot log history

# TODO: plot output


# 
# **Additional tasks**:
# Try different RNN cells (SimpleRNN and LSTM) and compare their performance/complexity trade-off.
# For comparison: try fitting a memoryless model (by just replacing the GRU layer) and see the difference.
# You could as well try-out a time-delay NN model. This will require slight changes the input/output data configuration.
# 
# Currently our RNN operates in stateless mode, i.e. the internal state / memory of the RNN is reset after each batch. 
# The mode can be changed by toggeling `stateful=True` for the RNN layer. However, this will require you to also declare the batch-size in the input layer. However, things aren't that simple unfortunately.
# 
# Additionally, the data needs to be re-arranged, since all streams in one batch are seen as parallel, independent time-series'. To ensure time continuity from one batch to another, the data needs to be arranged with (just listing the first dim here):\
# `[a1, b1, c1, d1, a2, b2, c2, d2, a3, ...] where the batch size is 4 in this example, thus there are four parallel streams and the number is the batch index. Thus, a1 needs to be time-contineous with a2, in order to make the stateful-flag work. 
# It is not mandatory to implement this to within this assignment - although it clearly limits the accuracy.
