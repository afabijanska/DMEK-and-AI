# -*- coding: utf-8 -*-
"""
@author: an_fab
"""

import os
import glob
import numpy as np
import configparser

from skimage import io
from skimage import color
from skimage import filters

from keras.models import model_from_json

#------------------------------------------------------------------------------
# read config
config = configparser.RawConfigParser()
config.read('configuration.txt')

data_dir = config.get('data_paths', 'data_dir_dmek')
copy_directory = config.get('data_paths', 'data_dir_dmek_copy')

num_patients =  int(config.get('image_params', 'num_dmek_patients'))

model_file = config.get('data_paths', 'model_file')
best_weights_file = config.get('data_paths', 'best_wetghts_file')

#######

model = model_from_json(open(model_file).read())
model.load_weights(best_weights_file)

# Iterate over the "Patient-i" directories 
for i in range(1, num_patients+1):
    
    patient_directory = os.path.join(data_dir, f"Patient-{i}")
    print(patient_directory )
    
    if os.path.exists(patient_directory):
        
        subdirectories = glob.glob(os.path.join(patient_directory, f"Patient-{i}*.month"))

        for subdirectory in subdirectories:
            
            # Construct the corresponding subdirectory path in the target copy directory
            copy_subdirectory = os.path.join(copy_directory, os.path.basename(subdirectory))

            # Create the target subdirectory if it doesn't exist
            os.makedirs(copy_subdirectory, exist_ok=True)
            
            files = glob.glob(os.path.join(subdirectory, "*.jpg"))
            
            for file in files:
                # Process each file as needed
                print(f"Reading file: {file}")

                target_file = os.path.join(copy_subdirectory, os.path.basename(file))
                
                img = io.imread(file)
                img = color.rgb2gray(img)
                
                X = img.shape[1]
                Y = img.shape[0]
                
                img = np.asarray(img, dtype='float16')
                img = np.reshape(img,(1, Y, X, 1))
                img = filters.gaussian(img)
                prediction = model.predict(img)
                prediction = prediction[:,:,:,1]
                
                prediction2 = np.reshape(prediction, (prediction.shape[1], prediction.shape[2]))
                prediction3 = 255*prediction2/np.max(prediction2)
                io.imsave(target_file, prediction3.astype(np.uint8))

                img = io.imread(file)
                img = 255*color.rgb2gray(img)
 
                t = 126
                bw = np.zeros(img.shape)
                bw[prediction3 > t] = 255
            
                img2 = color.gray2rgb(img)
                img2[bw == 255] = [0, 0, 255] 
                                           
                file2 = target_file.replace('.jpg','.png')
                io.imsave(file2, img2.astype(np.uint8))
                
    else:
        print(f"Patient-{i} directory not found.")