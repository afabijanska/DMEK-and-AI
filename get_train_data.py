# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 10:51:20 2024

@author: an_fab
"""

import os
import glob
import configparser

import numpy as np
import matplotlib.pyplot as plt

from skimage import io
from skimage import color
from skimage.filters import gaussian

from helpers import write_hdf5

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

num_patients = int(config.get('image_params', 'num_patients'))
num_controls = int(config.get('image_params', 'num_controls'))

data_dir = config.get('data_paths', 'data_dir_patients')
data_dir2 = config.get('data_paths', 'data_dir_controls')  
    
#if true adds Gaussian blur to the gt images prior training for regression
blur = eval(config.get('train_settings','blur'))

#------------------------------------------------------------------------------

def display_random(X_train, Y_train):
    
     indices = np.random.choice(X_train.shape[0], 5, replace=False)

     fig, axes = plt.subplots(2, 5, figsize=(15, 6))

     # Display the random images from X_train
     for i, ax in enumerate(axes[0]):    
         ax.imshow(X_train[indices[i]])
         ax.set_title(f"Image {indices[i]}")
         ax.axis('off')

     # Display the corresponding labels from Y_train in the second row
     for i, ax in enumerate(axes[1]):
         ax.imshow(Y_train[indices[i]])
         ax.axis('off')

     plt.show()
    
#------------------------------------------------------------------------------
# count the number of patient files

files_count = 0

for i in range(1, num_patients + 1):
    
     patient_directory = os.path.join(data_dir, f'Patient-{i}')
     
     if os.path.exists(patient_directory):
        
         subdirectories = glob.glob(os.path.join(patient_directory, f'Patient-{i}*.month'))
         
         for subdirectory in subdirectories:
                         
            subdirectory_bw = os.path.join(subdirectory, 'bw')
            
            files = glob.glob(os.path.join(subdirectory_bw, '*'))
            
            
            for file_name in files:
                
               print(file_name)
               files_count = files_count + 1
               
print('Files (patients) in total: ', files_count)               

#------------------------------------------------------------------------------
# count the number of control files

for i in range(1, num_controls + 1):
    
     control_directory = os.path.join(data_dir2, f'control-{i}')
     
     if os.path.exists(control_directory):
        
         subdirectory_bw = os.path.join(control_directory, 'bw')
            
         files = glob.glob(os.path.join(subdirectory_bw, '*'))
            
         for file_name in files:
                
              print(file_name)
              files_count = files_count + 1
               
#------------------------------------------------------------------------------
# get train data from annotated patients

X_train = np.zeros((files_count, Y, X), dtype = np.float16)
Y_train = np.zeros((files_count, Y, X), dtype = np.float16)

total_num_patches = 0

for i in range(1, num_patients + 1):
    
      patient_directory = os.path.join(data_dir, f'Patient-{i}')
     
      if os.path.exists(patient_directory):
        
          subdirectories = glob.glob(os.path.join(patient_directory, f'Patient-{i}*.month'))
         
          for subdirectory in subdirectories:
                         
            subdirectory_bw = os.path.join(subdirectory, 'bw')
            
            files = glob.glob(os.path.join(subdirectory_bw, '*'))
            
            for file_name in files:
                
                print(file_name)
               
                gt = io.imread(file_name)
               
                file_name_org = file_name.replace('bw','org')
                file_name_org = file_name_org.replace('.png','.jpg')
               
                img = io.imread(file_name_org)
                img = color.rgb2gray(img)
               
                print(file_name)
                print(file_name_org)
                
                X_train[total_num_patches] = img
                Y_train[total_num_patches] = gt
                       
                total_num_patches = total_num_patches + 1

#------------------------------------------------------------------------------
# get train data from annotated controls

for i in range(1, num_controls + 1):
    
      control_directory = os.path.join(data_dir2, f'control-{i}')
     
      if os.path.exists(control_directory):
                                 
          subdirectory_bw = os.path.join(control_directory, 'bw')
            
          files = glob.glob(os.path.join(subdirectory_bw, '*'))
            
          for file_name in files:
                
              print(file_name)
               
              gt = io.imread(file_name)
               
              file_name_org = file_name.replace('bw','org')
              file_name_org = file_name_org.replace('.png','.jpg')
               
              img = io.imread(file_name_org)
              img = color.rgb2gray(img)
               
              print(file_name)
              print(file_name_org)
                
              X_train[total_num_patches] = img
              Y_train[total_num_patches] = gt
                       
              total_num_patches = total_num_patches + 1

# #------------------------------------------------------------------------------

#X_train = X_train/255 #not needed, rgb2gray!
Y_train = Y_train//200

#------------------------------------------------------------------------------
# apply gaussian filter to binary mask if gaussian set true
                   
if blur:
    
    for i in range(0,  total_num_patches):
        
        img = gaussian(Y_train[i,:,:])
        Y_train[i,:,:] = img

#------------------------------------------------------------------------------

write_hdf5(X_train, file_X_train)
write_hdf5(Y_train, file_Y_train)

print(total_num_patches)

# display random train samples
display_random(X_train, Y_train)