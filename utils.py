import numpy as np
import nibabel as nib
import sys
import os

def load_nii(vol_name):
    X = nib.load(vol_name).get_data()
    X = np.reshape(X, X.shape + (1,))
    return X

