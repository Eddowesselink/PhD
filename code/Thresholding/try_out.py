import numpy as np 
import os
from glob import glob
import nibabel as nib
import seaborn as sns
sns.set_style("dark")
import pandas as pd
import seaborn as sns; sns.set()
from sklearn.cluster import KMeans
import scipy.stats as ss
import time 
from sklearn.mixture import GaussianMixture as GMM
import sys
from MuscleMap_codes import Thresholding as Thresh

#Here you can easily switch between Kmeans or GMM 
kmeans_activate = False
GMM_activate = True

# load dataset 
data_dir = 'D:/PS_Muscle_Segmentation/Data'

#Find all images and masks
all_images = glob(os.path.join(data_dir,'Training','*T2.nii.gz'))
all_masks = glob(os.path.join(data_dir,'Training','*GT.nii.gz'))

#Store all metrics in empty lists
muscle_multifidus_right = []
muscle_multifidus_left = []
muscle_erector_right = []
muscle_erector_left = []
muscle_psoas_right = []
muscle_psoas_left = []
fat_multifidus_right = []
fat_multifidus_left = []
fat_erector_right = []
fat_erector_left = []
fat_psoas_right = [] 
fat_psoas_left = []

volume_total_multifidus_right = []
volume_total_multifidus_left = []
volume_total_erector_right = []
volume_total_erector_left = []
volume_total_psoas_right = []
volume_total_psoas_left = []

volume_muscle_multifidus_right = []
volume_muscle_multifidus_left = []
volume_muscle_erector_right = []
volume_muscle_erector_left = []
volume_muscle_psoas_right = []
volume_muscle_psoas_left = []

volume_fat_multifidus_right = []
volume_fat_multifidus_left = []
volume_fat_erector_right = []
volume_fat_erector_left = []
volume_fat_psoas_right = []
volume_fat_psoas_left = []

#Save the threshold chosen to split muscle from fat
threshold_multifidus = []
threshold_erector = []
threshold_psoas = []

#Empty lists to save total time of model convergence
tik_tok_multifidus = []
tik_tok_erector = []
tik_tok_psoas = []
summary = []
ID = []
ID_name = []
header = []
header.append('Mean')
header.append('std')   

ii = 0

#Loop over training folder, and get all the images ending with T2.nii.gz
for y in glob(os.path.join(data_dir,'Training','*T2.nii.gz')): 
    print(all_masks[ii])
    #Load image and mask 
    img = nib.load(all_images[ii])
    mask = nib.load(all_masks[ii])

    #extract important header information
    sx,sy,sz = img.header['pixdim'][1:4]
    ID_name_file = all_images[ii][40:46]
    ID.append(ii + 1)
    ID_name.append(ID_name_file)

    #transfer image information to array
    img_array = img.get_fdata()
    img_array = np.array(img_array)

    #Load mask 
    mask_array = mask.get_fdata()
    mask_array = np.array(mask_array)

    #Update irator
    ii =  ii + 1
    
    #Here, we grab the corresponding labels for each of the muscles. 
    #It is important that the labelling is consistent with the default settings of the MuscleMap Community
    mask_multifidus = (mask_array == 1) | (mask_array == 2)
    mask_erector = (mask_array == 3) | (mask_array == 4)
    mask_psoas = (mask_array == 5) | (mask_array == 6)
    
    #get all voxels from masked image for each muscle (left and right combined) to calculate GMM and reshape to fit for unsupervised models
    mask_img_multifidus= np.reshape(img_array[mask_multifidus],(-1,1))
    mask_img_erector = np.reshape(img_array[mask_erector],(-1,1))
    mask_img_psoas = np.reshape(img_array[mask_psoas],(-1,1)) 
    
    thresholding = Thresh(mask_img_multifidus, mask_img_erector, mask_img_psoas, 2,'kmeans')
    
    

