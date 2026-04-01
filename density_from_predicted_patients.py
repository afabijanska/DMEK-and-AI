# -*- coding: utf-8 -*-
"""
@author: an_fab
"""

import re
import os
import glob
import configparser

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt 

from skimage import io
from skimage.color import rgb2gray
from skimage.measure import label   
from skimage.filters import median, sobel
from skimage import morphology

#-------------------------------------------------
def GetROI(img):
    
    img = rgb2gray(img)
    img = median(img)
    img = 255 * img
    bw = np.zeros(img.shape)
    
    bw[img > 30] = 1
    bw = getLargestCC(bw)
    
    grad_x = sobel(img, axis=0)  
    grad_y = sobel(img, axis=1)  
    
    magnitude = np.hypot(grad_x, grad_y) 
    print(magnitude.min(), magnitude.max())
    bw[magnitude<2] = 0
    
    bw = morphology.remove_small_objects(bw, min_size = 256)
    bw = median(bw, footprint = np.ones((15,15)))  
    bw = morphology.remove_small_holes(bw, area_threshold = 256)
      
    return bw

#-------------------------------------------------
# helper functions

def getLargestCC(segmentation):
    labels = label(segmentation)
    assert( labels.max() != 0 ) 
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC

#------------------------------------------------------------------------------
# read config
config = configparser.RawConfigParser()
config.read('configuration.txt')

#nubmer of test patients
num_test_patients =  int(config.get('image_params', 'num_dmek_patients'))

#image sizes in pixels
width_px = int(config.get('image_params','width'))
height_px = int(config.get('image_params','height'))

#image size in um
height_um = int(config.get('image_params','height_um')) 
width_um = int(config.get('image_params','width_um'))

#-------------------------------------------------
# data paths
data_dir = config.get('data_paths', 'data_dir_dmek')
fin_dir = config.get('data_paths', 'data_dir_dmek_fin')

# ------------------------------------------------
# pixel size
px_h = height_um / height_px
px_w = width_um / width_px

#-------------------------------------------------
# create dataframe to store the results

df = pd.DataFrame(columns = ['patient','month','side','centers','roi_area','density','case_id'])
all_data = []

#-------------------------------------------------
# read annotations and get densities

for i in range(1, num_test_patients + 1):
    
    patient_directory = os.path.join(data_dir, f"patient-{i}")
    
    copy_patient_dir = os.path.join(fin_dir, os.path.basename(patient_directory))
    
    # Create the target subdirectory if it doesn't exist
    os.makedirs(copy_patient_dir, exist_ok=True)
    
    if os.path.exists(patient_directory):
        
        subdirectories = glob.glob(os.path.join(patient_directory, f"patient-{i}*.month"))

        for subdirectory in subdirectories:
            
            # Construct the corresponding subdirectory path in the target copy directory
            copy_subdirectory = os.path.join(copy_patient_dir, os.path.basename(subdirectory))

            # Create the target subdirectory if it doesn't exist
            os.makedirs(copy_subdirectory, exist_ok = True)
            
            match = re.search(r'patient-(\d+) (right|left) (dmek|redmek|eye) (\d+)\.month', subdirectory)
        
            if match:
                print('##################')
                patient_num = int(match.group(1))
                side = match.group(2)
                dmek_type = match.group(3)
                month = int(match.group(4))
                print('patient: %d, side: %s, month: %d' % (patient_num, side, month))
                print('##################')
            
            files = glob.glob(os.path.join(subdirectory, "*.jpg"))
            
            patient_data = []
            
            for file in files:
                
                mask = np.zeros((height_px, width_px), dtype = np.uint8) 
                
                print(file) #original file
                
                img = io.imread(file)
                
                cell_file = file.replace('dmek\\','dmek3\\') #config copy directory
                cell_file = cell_file.replace('.jpg','.png')
                                
                roi = GetROI(img)
                roi2 = roi
                area = roi.sum() * px_h * px_w  #um^2
                area = area / 1000000  #mm^2
                mask[roi > 0] = 127
                
                cells = io.imread(cell_file)
                copy = cells
                cells = np.all(cells == [0, 0, 255], axis=-1).astype(np.uint8)
                cells = median(cells)
                mask[cells > 0] = 255
                
                centers = label(cells)
                num_centers = centers.max()
                density = num_centers/area
                
                print('\n%d, %f, %f' % (num_centers, area, density))
                
                filename = os.path.basename(file)
                new_row = [i, month, side, num_centers, area, density, filename]
                df.loc[len(df)] = new_row
                
                patient_data.append(density)
     
                file_fin = file.replace ('\\dmek\\','\\dmek_fin\\')
                io.imsave(file_fin, mask)
                
                img[cells == 1] = [0, 0, 255] 
                file_fin = file_fin.replace('.jpg','.png')
                io.imsave(file_fin, img)
                
                file_fin = file_fin.replace('.png', '_roi.png')
                io.imsave(file_fin, (255*roi2).astype(np.uint8))
                
        all_data.append(np.array(patient_data))
                
#-------------------------------------------------
# save to file
        
df.to_csv('cell_densities_patients.csv', sep = ';', index = False)

#----------------------------------------------------
# plot distributions 

# Derive unique months for each patient
unique_months_per_patient = df.groupby('patient')['month'].unique()

# Derive the overall month_order
month_order = sorted(set(month for months in unique_months_per_patient for month in months))

# Set the "month" column as an ordered categorical variable
df['month'] = pd.Categorical(df['month'], categories=month_order, ordered=True)

# Create subplots for each patient in a 4x6 matrix
unique_patients = df['patient'].unique()

nrows = 6
ncols = 3
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 18))
plt.rcParams.update({'font.size': 18})

pid = 1
for i, patient in enumerate(unique_patients):
    patient_data = df[df['patient'] == patient]

    row = i // ncols
    col = i % ncols

    # Plot boxplots for both sides in the same subplot
    sns.boxplot(x='month', y='density', hue='side', data=patient_data, ax=axes[row, col])
    axes[row, col].set_title(f'Patient {pid}')
    axes[row, col].set_xlabel('Month', fontsize=18)
    axes[row, col].set_ylabel('Density [cells/mm^2]', fontsize=18)
    pid = pid + 1 
    
plt.tight_layout()
plt.show()

#------
average_density = df.groupby(['patient', 'month', 'side'])['density'].mean().reset_index()
average_density = average_density.dropna()

median_density = df.groupby(['patient', 'month', 'side'])['density'].median().reset_index()
median_density = median_density.dropna()

std_density = df.groupby(['patient', 'month', 'side'])['density'].std().reset_index()
std_density = std_density.dropna()

# Merge df1 and df2
merged_df = pd.merge(average_density, median_density, on=['patient', 'month', 'side'], how='outer')

# Merge merged_df and df3
final_df = pd.merge(merged_df, std_density, on=['patient', 'month', 'side'], how='outer')
final_df.to_excel('average_densities.xlsx', index=False)