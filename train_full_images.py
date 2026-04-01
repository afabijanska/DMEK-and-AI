# -*- coding: utf-8 -*-
"""
@author: an_fab
"""

import configparser
import matplotlib.pyplot as plt 

from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

from helpers import load_hdf5
from AttUNet import AttUNetRegression, AttUNet_org

from skimage.filters import gaussian
#------------------------------------------------------------------------------
# read config

config = configparser.RawConfigParser()
config.read('configuration.txt')

#image sizes
X = int(config.get('image_params','width'))
Y = int(config.get('image_params','height'))

#hdf5 files with train images
file_X_train = config.get('data_paths', 'file_X_train')
file_Y_train = config.get('data_paths', 'file_Y_train')

#model and best weights file
model_file = config.get('data_paths', 'model_file')
best_weights_file = config.get('data_paths', 'best_wetghts_file')

#if true adds Gaussian blur to the gt images prior training for regression
blur = eval(config.get('train_settings','blur'))

#training settings
N_epochs = int(config.get('train_settings', 'num_epochs'))
batch_size = int(config.get('train_settings', 'batch_size'))
patience = int(config.get('train_settings', 'patience'))
batch_size = int(config.get('train_settings', 'batch_size'))
val_split = float(config.get('train_settings', 'val_split'))

#------------------------------------------------------------------------------
#load train data

X_train = load_hdf5(file_X_train)
patch_num = X_train.shape[0]
X_train = X_train.reshape((patch_num, Y, X, 1))

Y_train = load_hdf5(file_Y_train)
#Y_train = Y_train/127

#------------------------------------------------------------------------------
#define model

if blur:                                    #regression for gaussian blur
    i = 0
    for sample in Y_train:
        sample = gaussian(sample)
        Y_train[i] = sample
        i = i + 1
        
    model = AttUNetRegression(Y, X)
else:                                       #classification otherwise
    Y_train = to_categorical(Y_train)
    model = AttUNet_org(2, Y, X)

#save model to file
json_string = model.to_json()
open(model_file, 'w').write(json_string)

#------------------------------------------------------------------------------
#train model

#create callbacks for training
checkpointer = ModelCheckpoint(best_weights_file, 
                               verbose = 1, 
                               monitor = 'val_loss', 
                               mode = 'auto', 
                               save_best_only=True) #save at each epoch if the validation decreased

patienceCallBack = EarlyStopping(monitor = 'val_loss',
                                 patience = patience)

tbCallBack = TensorBoard(log_dir='./logs', 
                         histogram_freq = 0, 
                         write_graph = True, 
                         write_images = True, 
                         profile_batch = 100000000)

history = model.fit(X_train, 
                    Y_train, 
                    batch_size,
                    validation_split = val_split, 
                    verbose = 1, 
                    callbacks = [checkpointer,tbCallBack,patienceCallBack], 
                    shuffle = True, 
                    epochs = N_epochs)

model.save_weights('lastWeights.h5', overwrite=True)

#-----------------------------------------------------------------------------
#plot train history

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()