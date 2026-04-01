# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 13:11:44 2020

@author: an_fab
"""

import h5py

def load_hdf5(infile):
  with h5py.File(infile,"r") as f: 
    return f["image"][()]

def write_hdf5(arr,outfile):
  with h5py.File(outfile,"w") as f:
    f.create_dataset("image", data=arr, dtype=arr.dtype)