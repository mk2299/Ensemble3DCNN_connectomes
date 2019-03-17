from keras.layers import Conv3D, MaxPooling3D, Dropout,BatchNormalization, AveragePooling3D , Activation, Flatten, Dense
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.optimizers import Adagrad
from keras.optimizers import SGD
from keras import optimizers
from keras import backend as K
from keras import backend as be
import _pickle as cPickle
from keras import optimizers
from keras.utils.training_utils import multi_gpu_model
from keras import losses
from sklearn.metrics import mean_squared_error as mse
from keras import regularizers
import numpy as np
import keras

def Regressor3DCNN(input_size, output_classes=1):
    if len(input_size)!=4:
        raise ValueError('The input shape should have 4 dimensions with the last channel \
                         being the number of ROIs for the connectivity profile')
    model = Sequential()

    model.add(Conv3D(128, kernel_size=(3, 3, 3), input_shape= input_size,  border_mode='same'))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2), border_mode='same'))
    
    model.add(Conv3D(32, kernel_size=(3, 3, 3), border_mode='same'))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2), border_mode='same'))
    
    model.add(Flatten())
    model.add(Dense(32,activation='elu')) 
   
    model.add(Dense(output_classes, activation='linear'))
   
    return model 

    
def  Classifier3DCNN(input_size, output_classes=1):
    
    if len(input_size)!=4:
        raise ValueError('The input shape should have 4 dimensions with the last channel \
                         being the number of ROIs for the connectivity profile')
        
    model = Sequential()

    model.add(Conv3D(128, kernel_size=(3, 3, 3), input_shape=input_size,  border_mode='same'))
    model.add(Activation('elu'))
    model.add(Conv3D(32, kernel_size=(3, 3, 3), border_mode='same'))
    model.add(Activation('elu'))
    
    model.add(MaxPooling3D(pool_size=(2, 2, 2), border_mode='same'))
    
    model.add(Flatten())
    model.add(Dense(32, activation='elu', kernel_regularizer=regularizers.l2(0.005)))
    
    model.add(Dense(1, activation='sigmoid'))
    
    return model 



def Downsample(input_size):
    
    if len(input_size)!=4:
        raise ValueError('The input shape should have 4 dimensions with the last channel \
                         being the number of ROIs for the connectivity profile')
    model = Sequential()
    model.add(AveragePooling3D(pool_size=(2, 2, 2),input_shape = input_size, border_mode='same'))
    return model
    