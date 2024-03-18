#!/usr/bin/env python
# coding: utf-8

# # Autoencoder for MINST
# 
# In this assignment you will build, train and apply an autoencoder on the MINST dataset.
# Autoencoders are capable of finding a compressed format for an input/output. 
# If built with just a linear layer, they will essentially perform PCA, but with nonlinear hidden layers, they can perform additional nonlinear transformation, which might lead to a better compression ratio. 

# ### Setup
# As always, start with importing numpy, matplotlib and tensorflow as you are getting used to

# In[3]:


# TODO:
# ...
import tensorflow.keras as K


# ### Load training data
# We use the fashion part of the MNIST dataset, which is openly available. 
# Keras should download it for you automatically

# In[4]:


(X_train_full, y_train_full), _ = K.datasets.fashion_mnist.load_data()

# TODO: 
# rescale the grayscale images to have values to be within range 0 to 1
# split the training data into training and validation set by using the last 5000 images as validation set
# name them X_train and X_valid.


# A couple utility functions to plot the images:

# In[ ]:


def plot_image(image):
    plt.imshow(image, cmap="binary")
    plt.axis("off")

def show_reconstructions(model, images=X_valid, n_images=5):
    reconstructions = model.predict(images[:n_images])
    fig = plt.figure(figsize=(n_images * 1.5, 3))
    for image_index in range(n_images):
        plt.subplot(2, n_images, 1 + image_index)
        plot_image(images[image_index])
        plt.subplot(2, n_images, 1 + n_images + image_index)
        plot_image(reconstructions[image_index])
    plt.show()


# ### Build & Train the autoencoder

# We now want to build an autoencoder, e.g. two hidden layers in the encoder and respective decoder, that is able to retain B/W pixels.
# For the sizes of the layers, look at the dimension of your input as a reference. \
# Figure out the minimum size of the bottleneck such that the rounded_accuracy metric stays just above 93%.  
# 
# Check the dimension of your input images. To connect those to your dense layer, you will need to flatten the incoming data inside your model and when reconstructing, to reshape the data to match with the originals.
# 
# 
# Think about: 
# What will be the input and output of the autoencoder? \
# Which loss function is appropriate given the target metric? \
# Which activation is appropriate for the output of the decoder?

# In[ ]:


# pass this measure  as **metric** to your compile function. 
# Metrics are there to help you assess the progress/quality of your training, while not being actively involved in the minimization. 
# This could for example be a dB metric, instead of a 1.234e-4 MSE loss. 
def rounded_accuracy(y_true, y_pred):
    return K.metrics.binary_accuracy(tf.round(y_true), tf.round(y_pred))


# In[ ]:


# TODO:
#autoenc_model = 
#autoenc_model.compile(loss=, optimizer=, metrics=)
#hist = autoenc_model.fit(X=, Y=, epochs=10, validation_data=[])


# In[ ]:


# visualize the output alongside the original images
show_reconstructions(autoenc_model)


# - Try a linear-only auto encoder for reference. Was there really any benefit from having a NN here?
# - Evaluate how noise on the input data affects the outcome?
# - Find a way to tap into the model and extract the codings. Is there anything meaninful still left?
# - Optional: Try replacing some of the layers with convolutional/pooling layers (in the decoder it would need to be deconvolution). Does it affect the performance/complexity?
# 
# ### AI generated new fashion!
# Finally, have some fun and use the decoder as a generator for next-gen fashion by feeding it random codings ;) \
# While the result here might look quite crappy, this actually is a use case for the trained decoders: generating even more data. 

# In[ ]:


#TODO:

