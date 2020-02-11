from __future__ import print_function
import keras
print('Keras version : ', keras.__version__)
import tensorflow as tf
import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Dense, Input, Conv3D, MaxPooling3D, concatenate
from keras.optimizers import RMSprop

# deeper cnn model for mnist
from numpy import mean
from numpy import std
from matplotlib import pyplot
from keras.utils import to_categorical
from keras.layers import Flatten
from keras.optimizers import SGD

import matplotlib.pyplot as plt
import numpy as np
#############################################
############## Make the model ###############
#############################################


def make_one_branch_model(temporal_dim, width, height, channels, nb_class):
    #TODO
    #Build the 'one branch' model and compile it.
    
    # model building
    input_data = Input(shape=(temporal_dim, width, height, channels))
    #convolutional layer with rectified linear activation unit
    output = Conv3D(30, kernel_size=(3,3,3), padding='same', activation='relu')(input_data)
    #30 convolution filters used each of size 3x3
    
    #choose the best featurs used via pooling
    output = MaxPooling3D(pool_size=(2,2,2))(output)
    
    
    
    output = Conv3D(60, kernel_size=(3,3,3), padding='same', activation='relu')(output)
    #60 convolution filters used each of size 3x3
    output = MaxPooling3D(pool_size=(2,2,2))(output)
    #choose the best featurs used via pooling
    
    output = Conv3D(80, kernel_size=(3,3,3), padding='same', activation='relu')(output)
    #80 convolution filters used each of size 3x3
    
    #choose the best featurs used via pooling
    output = MaxPooling3D(pool_size=(2,2,2))(output)
    #flatten since too many dimensions, but  we only want a classfication output
    output = Flatten() (output)
    
    #fully connected to get all output data
    output = Dense(500, activation ='relu')(output)
    output = Dense(nb_class, activation ='softmax')(output)
    
    model = Model(inputs = input_data, outputs = output)
    sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.5, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics = ['accuracy'])
    return model

def make_branch_model(temporal_dim, width, height, channels, nb_class):
    #TODO
    #Build the 'one branch' model and compile it.
    
    # model building
    input_data = Input(shape=(temporal_dim, width, height, channels))
    #convolutional layer with rectified linear activation unit
    output = Conv3D(30, kernel_size=(3,3,3), padding='same', activation='relu')(input_data)
    #30 convolution filters used each of size 3x3
    
    #choose the best featurs used via pooling
    output = MaxPooling3D(pool_size=(2,2,2))(output)
    
    output = Conv3D(60, kernel_size=(3,3,3), padding='same', activation='relu')(output)
    #60 convolution filters used each of size 3x3
    output = MaxPooling3D(pool_size=(2,2,2))(output)
    #choose the best featurs used via pooling
    
    output = Conv3D(80, kernel_size=(3,3,3), padding='same', activation='relu')(output)
    #80 convolution filters used each of size 3x3
    
    #choose the best featurs used via pooling
    output = MaxPooling3D(pool_size=(2,2,2))(output)
    #flatten since too many dimensions, but  we only want a classfication output
    output = Flatten() (output)
    
    #fully connected to get all output data
    output = Dense(500, activation ='relu')(output)
    return output, input_data


def make_model(temporal_dim, width, height, nb_class):
    #TODO
    #Build the siamese model and compile it.
    #Use the following optimizer
    sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.5, nesterov=True)
    RGBModel, RGBinput = make_branch_model(temporal_dim, width, height, 3, nb_class)
    FlowModel, Flowinput = make_branch_model(temporal_dim, width, height, 2, nb_class)
    
    siamese = concatenate([RGBModel, FlowModel])
    
    #siamese = Dense(nb_class, activation = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.5, nesterov=True))(siamese)
    
    
    siamese = Dense(nb_class, activation ='softmax')(siamese)
    
    model = Model(inputs = [RGBinput, Flowinput], outputs = siamese)
    
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics = ['accuracy'])
    return model
    




