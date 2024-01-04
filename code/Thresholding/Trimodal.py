import numpy as np 
import matplotlib.pyplot as plt
import os
from glob import glob
import nibabel as nib
from sklearn.mixture import GaussianMixture as GMM
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set_style("dark")
import pandas as pd
import seaborn as sns; sns.set()
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from collections import Counter, defaultdict
from sklearn.decomposition import PCA
import numpy.ma as ma
import scipy.stats as ss
from sklearn.cluster import kmeans_plusplus
import time

kmeans = False
GMM_on = True

# load dataset 
data_dir = 'D:/PS_Muscle_Segmentation/Data'

#Find all images and masks
all_images = glob(os.path.join(data_dir,'Training','*T2.nii.gz'))
all_masks = glob(os.path.join(data_dir,'Training','*GT.nii.gz'))

muscle_multifidus_right = []
muscle_multifidus_left = []
muscle_erector_right = [] 
muscle_erector_left = []
muscle_psoas_right = []
muscle_psoas_left = []

undefined_multifidus_right = []
undefined_multifidus_left = []
undefined_erector_right = []
undefined_erector_left = []
undefined_psoas_right = []
undefined_psoas_left = []

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

volume_undefined_multifidus_right = []
volume_undefined_multifidus_left = []
volume_undefined_erector_right = []
volume_undefined_erector_left = []
volume_undefined_psoas_right = []
volume_undefined_psoas_left = []

volume_fat_multifidus_right = []
volume_fat_multifidus_left = []
volume_fat_erector_right = []
volume_fat_erector_left = []
volume_fat_psoas_right = []
volume_fat_psoas_left = []

Threshold_multifidus_muscle_UD = [] 
Threshold_multifidus_UD_fat = []

Threshold_erector_muscle_UD = [] 
Threshold_erector_UD_fat = []

Threshold_psoas_muscle_UD = [] 
Threshold_psoas_UD_fat = []

tik_tok_GMM = []
tik_tok_kmeans = []

tik_tok_multifidus = []
tik_tok_erector = []
tik_tok_psoas = []

ID = []
ID_name = []
    
iii = 0
ii = 0
for y in glob(os.path.join(data_dir,'Training','*T2.nii.gz')): 
      
    #Load image load 
    img = nib.load(all_images[ii])
    img_array = img.get_fdata()
    sx,sy,sz = img.header['pixdim'][1:4]
    img_array = np.array(img_array)
    ID_full = all_images[ii]
    ID_name_file = ID_full[40:46]
    ID.append(ii + 1)
    ID_name.append(ID_name_file)
    print(ID_name_file)
    
    #Load mask
    mask = nib.load(all_masks[ii])
    min_pixdim=np.min(mask.header['pixdim'][1:4])
    mask_array = mask.get_fdata()
    mask_array = np.array(mask_array)
    ii =  ii + 1
    
    mask_multifidus1 = mask_array == 1
    mask_multifidus2 = mask_array == 2
    mask_multifidus = mask_multifidus1 + mask_multifidus2
    
    mask_erector1 = mask_array == 3
    mask_erector2 = mask_array == 4
    
    mask_erector = mask_erector1 + mask_erector2
    
    mask_psoas1 = mask_array == 5
    mask_psoas2 = mask_array == 6
    mask_psoas = mask_psoas1 + mask_psoas2 
    
    #get all voxels per muscles combined to calculate GMM
    mask_img_multifidus= img_array[mask_multifidus]
    mask_img_erector = img_array[mask_erector]
    mask_img_psoas = img_array[mask_psoas]
    
    B = np.reshape(mask_img_multifidus, (-1, 1))
    D = np.reshape(mask_img_erector, (-1, 1))
    C = np.reshape(mask_img_psoas, (-1, 1))   
    
    if kmeans:
        kmeans = KMeans(n_clusters = 3, init = 'k-means++', tol = 0.001, n_init = 50, max_iter = 1000)
        
        start = time.time()
        labels_multifidus = kmeans.fit(B).predict(B)
        stop = time.time()
        tik_tok_multifidus.append(stop - start)
        
        start = time.time()
        labels_erector = kmeans.fit(D).predict(D)
        stop = time.time()
        tik_tok_erector.append(stop - start)
        
        start = time.time()
        labels_psoas = kmeans.fit(C).predict(C)
        stop = time.time()
        tik_tok_psoas.append(stop - start)
    
    if GMM_on:
        gmm = GMM(n_components = 3, covariance_type= 'full', n_init = 50,  init_params = 'kmeans', tol=(0.001), max_iter = 1000)

        start = time.time()
        labels_multifidus = gmm.fit(B).predict(B)
        stop = time.time()
        tik_tok_multifidus.append(stop - start)
        
        start = time.time()
        labels_erector = gmm.fit(D).predict(D)
        stop = time.time()
        tik_tok_erector.append(stop - start)
        
        start = time.time()
        labels_psoas = gmm.fit(C).predict(C)
        stop = time.time()
        tik_tok_psoas.append(stop - start)
    
    mask_muscle_multifidus1 = labels_multifidus == 0
    mask_muscle_multifidus3 = mask_img_multifidus[mask_muscle_multifidus1]
    mean_mask_muscle_multifidus  = np.mean(mask_muscle_multifidus3)
    
    mask_undefined_multifidus1 = labels_multifidus == 1
    mask_undefined_multifidus3 = mask_img_multifidus [mask_undefined_multifidus1]
    mean_undefined_multifidus  = np.mean(mask_undefined_multifidus3)
    
    mask_fat_multifidus1 = labels_multifidus == 2
    mask_fat_multifidus3 =  mask_img_multifidus[mask_fat_multifidus1]
    mean_mask_fat_multifidus = np.mean(mask_fat_multifidus3)

    muscle_fixed_multifidus = mask_muscle_multifidus3
    undefined_fixed_multifidus = mask_undefined_multifidus3
    fat_fixed_multifidus = mask_fat_multifidus3

    idxmean = np.array([mean_mask_muscle_multifidus, mean_undefined_multifidus,mean_mask_fat_multifidus])
    idxmean1 = ss.rankdata (idxmean, method ='min')
    
    if idxmean1[0] == 2 and idxmean1[1] == 1 and idxmean1[2] == 3:
        mask_muscle_multifidus3  = undefined_fixed_multifidus
        mask_undefined_multifidus3 = muscle_fixed_multifidus 
    elif idxmean1[0] == 3 and idxmean1[1] == 2 and idxmean1[2] == 1:
        mask_muscle_multifidus3  = fat_fixed_multifidus
        mask_fat_multifidus3 = muscle_fixed_multifidus 
    elif idxmean1[0] == 3 and idxmean1[1] == 1 and idxmean1[2] == 2:
        mask_muscle_multifidus3  = undefined_fixed_multifidus
        mask_undefined_multifidus3 = fat_fixed_multifidus
        mask_fat_multifidus3 = muscle_fixed_multifidus 
    elif idxmean1[0] == 2 and idxmean1[1] == 3 and idxmean1[2] == 1:
        mask_muscle_multifidus3  = fat_fixed_multifidus
        mask_undefined_multifidus3 = muscle_fixed_multifidus 
        mask_fat_multifidus3 = undefined_fixed_multifidus
    elif idxmean1[0] == 1 and idxmean1[1] == 2 and idxmean1[2] == 3:
        mask_muscle_multifidus3  = muscle_fixed_multifidus 
        mask_undefined_multifidus3 = undefined_fixed_multifidus
        mask_fat_multifidus3 = fat_fixed_multifidus
    elif idxmean1[0] == 1 and idxmean1[1] == 3 and idxmean1[2] == 2:
        mask_undefined_multifidus3 = fat_fixed_multifidus
        mask_fat_multifidus3 = undefined_fixed_multifidus
        
    muscle_multifidus_upper = max(mask_muscle_multifidus3)
    print('max muscle multifidus',muscle_multifidus_upper)
    muscle_multifidus_lower = min(mask_muscle_multifidus3)
    print('min muscle multifidus', muscle_multifidus_lower)
    
    undefined_multifidus_upper = max(mask_undefined_multifidus3)
    print('max undefined multifidus',undefined_multifidus_upper)
    undefined_multifidus_lower = min(mask_undefined_multifidus3)
    print('min undefined multifidus', undefined_multifidus_lower)
    
    fat_multifidus_upper = max(mask_fat_multifidus3)
    print('max fat multifidus',fat_multifidus_upper)
    fat_multifidus_lower = min(mask_fat_multifidus3)
    print('min fat multifidus', fat_multifidus_lower)
    
    undefined_multifidus_lower1= undefined_multifidus_lower
    muscle_multifidus_upper1 = muscle_multifidus_upper
    
    if undefined_multifidus_lower <= muscle_multifidus_lower:
        undefined_multifidus_lower = muscle_multifidus_upper1
        muscle_multifidus_lower = undefined_multifidus_lower1
    if fat_multifidus_lower <= undefined_multifidus_lower:
        fat_multifidus_lower = undefined_multifidus_upper
    if fat_multifidus_lower <= muscle_multifidus_lower:
        fat_multifidus_lower = undefined_multifidus_upper
        
    Threshold_multifidus_muscle_UD.append(muscle_multifidus_upper)
    Threshold_multifidus_UD_fat.append (fat_multifidus_lower)
    
    mask_muscle_erector1 = labels_erector == 0
    mask_muscle_erector3 = mask_img_erector[mask_muscle_erector1]
    mean_mask_muscle_erector  = np.mean(mask_muscle_erector3)
    
    mask_undefined_erector1 = labels_erector == 1
    mask_undefined_erector3 = mask_img_erector[mask_undefined_erector1]
    mean_undefined_undefined_erector  = np.mean(mask_undefined_erector3)
    
    mask_fat_erector1 = labels_erector == 2
    mask_fat_erector3=  mask_img_erector[mask_fat_erector1]
    mean_mask_fat_erector = np.mean(mask_fat_erector3)

    muscle_fixed_erector = mask_muscle_erector3
    undefined_fixed_erector = mask_undefined_erector3
    fat_fixed_erector = mask_fat_erector3

    idxmean = np.array([mean_mask_muscle_erector, mean_undefined_undefined_erector,mean_mask_fat_erector])
    idxmean1 = ss.rankdata (idxmean, method ='min')
    
    if idxmean1[0] == 2 and idxmean1[1] == 1 and idxmean1[2] == 3:
        mask_muscle_erector3 = undefined_fixed_erector
        mask_undefined_erector3 = muscle_fixed_erector 
    elif idxmean1[0] == 3 and idxmean1[1] == 2 and idxmean1[2] == 1:
        mask_muscle_erector3  = fat_fixed_erector
        mask_fat_erector3 = muscle_fixed_erector 
    elif idxmean1[0] == 3 and idxmean1[1] == 1 and idxmean1[2] == 2:
        mask_muscle_erector3  = undefined_fixed_erector
        mask_undefined_erector3 = fat_fixed_erector
        mask_fat_erector3 = muscle_fixed_erector
    elif idxmean1[0] == 2 and idxmean1[1] == 3 and idxmean1[2] == 1:
        mask_muscle_erector3 = fat_fixed_erector
        mask_undefined_erector3 = muscle_fixed_erector
        mask_fat_erector3 = undefined_fixed_erector
    elif idxmean1[0] == 1 and idxmean1[1] == 2 and idxmean1[2] == 3:
        mask_muscle_erector3 = muscle_fixed_erector 
        mask_undefined_erector3= undefined_fixed_erector
        mask_fat_erector3 = fat_fixed_erector
    elif idxmean1[0] == 1 and idxmean1[1] == 3 and idxmean1[2] == 2:
        mask_undefined_erector3 = fat_fixed_erector
        mask_fat_erector3 = undefined_fixed_erector
        
    muscle_erector_upper = max(mask_muscle_erector3)
    print('max muscle erector',muscle_erector_upper)
    muscle_erector_lower = min(mask_muscle_erector3)
    print('min muscle erector', muscle_erector_lower)
    
    undefined_erector_upper = max(mask_undefined_erector3)
    print('max undefined erector',undefined_erector_upper)
    undefined_erector_lower = min(mask_undefined_erector3)
    print('min undefined erector', undefined_erector_lower)
    
    fat_erector_upper = max(mask_fat_erector3)
    print('max fat erector',fat_erector_upper)
    fat_erector_lower = min(mask_fat_erector3)
    print('min fat erector', fat_erector_lower)
    
    undefined_erector_lower1= undefined_erector_lower
    muscle_erector_upper1 = muscle_erector_upper
    
    if undefined_erector_lower <= muscle_erector_lower:
        undefined_erector_lower = muscle_erector_upper1
        muscle_erector_lower = undefined_erector_lower1
    if fat_erector_lower <= undefined_erector_lower :
        fat_erector_lower = undefined_erector_upper
    if fat_erector_lower <= muscle_erector_lower :
        fat_erector_lower = undefined_erector_upper

    Threshold_erector_muscle_UD.append(muscle_erector_upper)
    Threshold_erector_UD_fat.append(fat_erector_lower)

    mask_muscle_psoas1 = labels_psoas == 0
    mask_muscle_psoas3 = mask_img_psoas[mask_muscle_psoas1]
    mean_mask_muscle_psoas = np.mean(mask_muscle_psoas3)
    
    mask_undefined_psoas1 = labels_psoas== 1
    mask_undefined_psoas3 =  mask_img_psoas[mask_undefined_psoas1] 
    mean_undefined_undefined_psoas  = np.mean(mask_undefined_psoas3)
    
    mask_fat_psoas1 = labels_psoas == 2
    mask_fat_psoas3 = mask_img_psoas[mask_fat_psoas1]  
    mean_mask_fat_psoas = np.mean(mask_fat_psoas3)

    muscle_fixed_psoas = mask_muscle_psoas3
    undefined_fixed_psoas = mask_undefined_psoas3
    fat_fixed_psoas = mask_fat_psoas3

    idxmean = np.array([mean_mask_muscle_psoas, mean_undefined_undefined_psoas,mean_mask_fat_psoas])
    idxmean1 = ss.rankdata (idxmean, method ='min')
    
    if idxmean1[0] == 2 and idxmean1[1] == 1 and idxmean1[2] == 3:
        mask_muscle_psoas3 = undefined_fixed_psoas
        mask_undefined_psoas3 = muscle_fixed_psoas
    elif idxmean1[0] == 3 and idxmean1[1] == 2 and idxmean1[2] == 1:
        mask_muscle_psoas3  = fat_fixed_psoas
        mask_fat_psoas3 = muscle_fixed_psoas 
    elif idxmean1[0] == 3 and idxmean1[1] == 1 and idxmean1[2] == 2:
        mask_muscle_psoas3  = undefined_fixed_psoas
        mask_undefined_psoas3 = fat_fixed_psoas
        mask_fat_psoas3 = muscle_fixed_psoas
    elif idxmean1[0] == 2 and idxmean1[1] == 3 and idxmean1[2] == 1:
        mask_muscle_psoas3 = fat_fixed_psoas
        mask_undefined_psoas3 = muscle_fixed_psoas
        mask_fat_psoas3 = undefined_fixed_psoas
    elif idxmean1[0] == 1 and idxmean1[1] == 2 and idxmean1[2] == 3:
        mask_muscle_psoas3 = muscle_fixed_psoas 
        mask_undefined_psoas3= undefined_fixed_psoas
        mask_fat_psoas3 = fat_fixed_psoas
    elif idxmean1[0] == 1 and idxmean1[1] == 3 and idxmean1[2] == 2:
        mask_undefined_psoas3 = fat_fixed_psoas
        mask_fat_psoas3 = undefined_fixed_psoas
        
    muscle_psoas_upper = max(mask_muscle_psoas3)
    print('max muscle psoas',muscle_psoas_upper)
    muscle_psoas_lower = min(mask_muscle_psoas3)
    print('min muscle psoas', muscle_psoas_lower)
    
    undefined_psoas_upper = max(mask_undefined_psoas3)
    print('max undefined psoas',undefined_psoas_upper)
    undefined_psoas_lower = min(mask_undefined_psoas3)
    print('min undefined psoas', undefined_psoas_lower)
    
    fat_psoas_upper = max(mask_fat_psoas3)
    print('max fat psoas',fat_psoas_upper)
    fat_psoas_lower = min(mask_fat_psoas3)
    print('min fat psoas', fat_psoas_lower)
    
    undefined_psoas_lower1= undefined_psoas_lower
    muscle_psoas_upper1 = muscle_psoas_upper
    
    if undefined_psoas_lower <= muscle_psoas_lower:
        undefined_psoas_lower = muscle_psoas_upper1
        muscle_psoas_lower = undefined_psoas_lower1
    if fat_psoas_lower <= undefined_psoas_lower:
         fat_psoas_lower = undefined_psoas_upper
    if fat_psoas_lower <= muscle_psoas_lower:
        fat_psoas_lower = undefined_psoas_upper

    Threshold_psoas_muscle_UD.append(muscle_psoas_upper)
    Threshold_psoas_UD_fat.append(fat_psoas_lower)

    number_of_labels = mask_array.max()
    labelname_1 = 'Multifidus_right'
    labelname_2 = 'Multifidus_left'
    labelname_3 = 'Erector_spinae_right'
    labelname_4 = 'Erector_spinae_left'
    labelname_5 = 'Psoas_right'
    labelname_6 = 'Psoas_left'
        
    for label in range(1,int(number_of_labels)+1):
        
        if label == 1:
            print(labelname_1)
            mask_label = mask_array == label
            mask_img2 = mask_label * img_array
            mask_img3 = img_array[mask_label]
            
            muscle_img_label = mask_img3 <= muscle_multifidus_upper 
            
            undefined_img_label1 = mask_img3 > muscle_multifidus_upper 
            undefined_img_label2 = mask_img3 < fat_multifidus_lower
            undefined_img_label = undefined_img_label1 * undefined_img_label2
            
            fat_img_label = mask_img3 >= fat_multifidus_lower
            
            total = np.sum(muscle_img_label) + np.sum(undefined_img_label) + np.sum(fat_img_label)
            
            volume_total = ((sx * sy * sz)* total) / 1000
            muscle_total = np.sum(muscle_img_label)
            undefined_total = np.sum(undefined_img_label)
            fat_total = np.sum(fat_img_label)
            
            
            volume_muscle = ((sx * sy * sz)* muscle_total) / 1000
            volume_undefined = ((sx * sy * sz)* undefined_total) / 1000
            volume_fat = ((sx * sy * sz)* fat_total) / 1000
            volume_muscle_multifidus_right.append(volume_muscle)
            volume_undefined_multifidus_right.append(volume_undefined)
            volume_fat_multifidus_right.append(volume_fat)
            muscle_multifidus_right.append((muscle_total / total) * 100 )
            undefined_multifidus_right.append((undefined_total / total) * 100 )
            fat_multifidus_right.append((fat_total / total) * 100)
            volume_total_multifidus_right.append(volume_total)
            
            print((muscle_total / total) * 100 )
            print((undefined_total/total)*100)
            print((fat_total / total) * 100)
        
            mask_muscle = mask_img2 <= muscle_multifidus_upper 
            mask_muscle = mask_muscle * mask_label
            mask_muscle = mask_muscle > 0
            mask_muscle = mask_muscle * 10
            
            muscle_mask_multifidus_right =  mask_muscle.reshape(img_array.shape)
            
            mask_undefined = mask_img2 > muscle_multifidus_upper 
            mask_undefined = mask_undefined * mask_label
            mask_undefined = mask_img2 < fat_multifidus_lower 
            mask_undefined = mask_undefined * mask_label
            mask_undefined = mask_undefined > 0
            mask_undefined = mask_undefined* 50
            
            undefined_mask_multifidus_right =  mask_undefined.reshape(img_array.shape)
              
            fat_img = mask_img2 >= fat_multifidus_lower
            fat_img = fat_img * mask_label
            fat_img = fat_img > 0
            fat_img = fat_img * 100    
            
            fat_mask_multifidus_right = fat_img.reshape(img_array.shape)
            
        if label == 2:
            print(labelname_2)
            mask_label = mask_array == label
            mask_img2 = mask_label * img_array
            mask_img3 = img_array[mask_label]
            
            muscle_img_label = mask_img3 <= muscle_multifidus_upper 
            
            undefined_img_label1 = mask_img3 > muscle_multifidus_upper 
            undefined_img_label2 = mask_img3 < fat_multifidus_lower
            undefined_img_label = undefined_img_label1 * undefined_img_label2
            
            fat_img_label = mask_img3 >= fat_multifidus_lower
            
            total = np.sum(muscle_img_label) + np.sum(undefined_img_label) + np.sum(fat_img_label)
            
            volume_total = ((sx * sy * sz)* total) / 1000
            muscle_total = np.sum(muscle_img_label)
            undefined_total = np.sum(undefined_img_label)
            fat_total = np.sum(fat_img_label)
            
            volume_muscle = ((sx * sy * sz)* muscle_total) / 1000
            volume_undefined = ((sx * sy * sz)* undefined_total) / 1000
            volume_fat = ((sx * sy * sz)* fat_total) / 1000
            volume_muscle_multifidus_left.append(volume_muscle)
            volume_undefined_multifidus_left.append(volume_undefined)
            volume_fat_multifidus_left.append(volume_fat)
            muscle_multifidus_left.append((muscle_total / total) * 100 )
            undefined_multifidus_left.append((undefined_total / total) * 100 )
            fat_multifidus_left.append((fat_total / total) * 100)
            volume_total_multifidus_left.append(volume_total)
            
            print((muscle_total / total) * 100 )
            print((undefined_total/total)*100)
            print((fat_total / total) * 100)
        
            mask_muscle = mask_img2 <= muscle_multifidus_upper 
            mask_muscle = mask_muscle * mask_label
            mask_muscle = mask_muscle > 0
            mask_muscle = mask_muscle * 10
            
            muscle_mask_multifidus_left =  mask_muscle.reshape(img_array.shape)
            
            mask_undefined = mask_img2 > muscle_multifidus_upper 
            mask_undefined = mask_undefined * mask_label
            mask_undefined = mask_img2 < fat_multifidus_lower 
            mask_undefined = mask_undefined * mask_label
            mask_undefined = mask_undefined > 0
            mask_undefined = mask_undefined* 50
            
            undefined_mask_multifidus_left=  mask_undefined.reshape(img_array.shape)
              
            fat_img = mask_img2 >= fat_multifidus_lower
            fat_img = fat_img * mask_label
            fat_img = fat_img > 0
            fat_img = fat_img * 100        
            
            fat_mask_multifidus_left = fat_img.reshape(img_array.shape)
            
        if label == 3:
            print(labelname_3)
            mask_label = mask_array == label
            mask_img2 = mask_label * img_array
            mask_img3 = img_array[mask_label]
            
            muscle_img_label = mask_img3 <= muscle_erector_upper 
            
            undefined_img_label1 = mask_img3 > muscle_erector_upper 
            undefined_img_label2 = mask_img3 < fat_erector_lower
            undefined_img_label = undefined_img_label1 * undefined_img_label2
            
            fat_img_label = mask_img3 >= fat_erector_lower 
            
            total = np.sum(muscle_img_label) + np.sum(undefined_img_label) + np.sum(fat_img_label)
            volume_total = ((sx * sy * sz)* total) / 1000
            muscle_total = np.sum(muscle_img_label)
            undefined_total = np.sum(undefined_img_label)
            fat_total = np.sum(fat_img_label)
            volume_muscle = ((sx * sy * sz)* muscle_total) / 1000
            volume_undefined = ((sx * sy * sz)* undefined_total) / 1000
            volume_fat = ((sx * sy * sz)* fat_total) / 1000
            volume_muscle_erector_right.append(volume_muscle)
            volume_undefined_erector_right.append(volume_undefined)
            volume_fat_erector_right.append(volume_fat)
            muscle_erector_right.append((muscle_total / total) * 100 )
            undefined_erector_right.append((undefined_total / total) * 100 )
            fat_erector_right.append((fat_total / total) * 100)
            volume_total_erector_right.append(volume_total)
            
            print((muscle_total / total) * 100 )
            print((undefined_total/total)*100)
            print((fat_total / total) * 100)
        
            mask_muscle = mask_img2 <= muscle_erector_upper 
            mask_muscle = mask_muscle * mask_label
            mask_muscle = mask_muscle > 0
            mask_muscle = mask_muscle * 10
            
            muscle_mask_erector_right =  mask_muscle.reshape(img_array.shape)
            
            mask_undefined = mask_img2 > muscle_erector_upper 
            mask_undefined = mask_undefined * mask_label
            mask_undefined = mask_img2 < fat_erector_lower 
            mask_undefined = mask_undefined * mask_label
            mask_undefined = mask_undefined > 0
            mask_undefined = mask_undefined* 50
            
            undefined_mask_erector_right=  mask_undefined.reshape(img_array.shape)
              
            fat_img = mask_img2 >= fat_erector_lower
            fat_img = fat_img * mask_label
            fat_img = fat_img > 0
            fat_img = fat_img * 100        
            
            fat_mask_erector_right = fat_img.reshape(img_array.shape)
            
        if label == 4:
            print(labelname_4)
            mask_label = mask_array == label
            mask_img2 = mask_label * img_array
            mask_img3 = img_array[mask_label]
            
            muscle_img_label = mask_img3 <= muscle_erector_upper 
            
            undefined_img_label1 = mask_img3 > muscle_erector_upper 
            undefined_img_label2 = mask_img3 < fat_erector_lower
            undefined_img_label = undefined_img_label1 * undefined_img_label2
            
            fat_img_label = mask_img3 >= fat_erector_lower 
            
            total = np.sum(muscle_img_label) + np.sum(undefined_img_label) + np.sum(fat_img_label)
            volume_total = ((sx * sy * sz)* total) / 1000
            muscle_total = np.sum(muscle_img_label)
            undefined_total = np.sum(undefined_img_label)
            fat_total = np.sum(fat_img_label)
            volume_muscle = ((sx * sy * sz)* muscle_total) / 1000
            volume_undefined = ((sx * sy * sz)* undefined_total) / 1000
            volume_fat = ((sx * sy * sz)* fat_total) / 1000
            volume_muscle_erector_left.append(volume_muscle)
            volume_undefined_erector_left.append(volume_undefined)
            volume_fat_erector_left.append(volume_fat)
            muscle_erector_left.append((muscle_total / total) * 100 )
            undefined_erector_left.append((undefined_total / total) * 100 )
            fat_erector_left.append((fat_total / total) * 100)
            volume_total_erector_left.append(volume_total)
            
            print((muscle_total / total) * 100 )
            print((undefined_total/total)*100)
            print((fat_total / total) * 100)
        
            mask_muscle = mask_img2 <= muscle_erector_upper 
            mask_muscle = mask_muscle * mask_label
            mask_muscle = mask_muscle > 0
            mask_muscle = mask_muscle * 10
            
            muscle_mask_erector_left =  mask_muscle.reshape(img_array.shape)
            
            mask_undefined = mask_img2 > muscle_erector_upper 
            mask_undefined = mask_undefined * mask_label
            mask_undefined = mask_img2 < fat_erector_lower 
            mask_undefined = mask_undefined * mask_label
            mask_undefined = mask_undefined > 0
            mask_undefined = mask_undefined* 50
            
            undefined_mask_erector_left=  mask_undefined.reshape(img_array.shape)
              
            fat_img = mask_img2 >= fat_erector_lower
            fat_img = fat_img * mask_label
            fat_img = fat_img > 0
            fat_img = fat_img * 100       
            
            fat_mask_erector_left = fat_img.reshape(img_array.shape)
            
            
        if label == 5:
            print(labelname_5)
            mask_label = mask_array == label
            mask_img2 = mask_label * img_array
            mask_img3 = img_array[mask_label]
            
            muscle_img_label = mask_img3 <= muscle_psoas_upper 
            
            undefined_img_label1 = mask_img3 > muscle_psoas_upper 
            undefined_img_label2 = mask_img3 < fat_psoas_lower
            undefined_img_label = undefined_img_label1 * undefined_img_label2
            
            fat_img_label = mask_img3 >= fat_psoas_lower 
            
            total = np.sum(muscle_img_label) + np.sum(undefined_img_label) + np.sum(fat_img_label)
            volume_total = ((sx * sy * sz)* total) / 1000
            muscle_total = np.sum(muscle_img_label)
            undefined_total = np.sum(undefined_img_label)
            fat_total = np.sum(fat_img_label)
            volume_muscle = ((sx * sy * sz)* muscle_total) / 1000
            volume_undefined = ((sx * sy * sz)* undefined_total) / 1000
            volume_fat = ((sx * sy * sz)* fat_total) / 1000
            volume_muscle_psoas_right.append(volume_muscle)
            volume_undefined_psoas_right.append(volume_undefined)
            volume_fat_psoas_right.append(volume_fat)
            muscle_psoas_right.append((muscle_total / total) * 100 )
            undefined_psoas_right.append((undefined_total / total) * 100 )
            fat_psoas_right.append((fat_total / total) * 100)
            volume_total_psoas_right.append(volume_total)
            
            print((muscle_total / total) * 100 )
            print((undefined_total/total)*100)
            print((fat_total / total) * 100)
        
            mask_muscle = mask_img2 <= muscle_psoas_upper 
            mask_muscle = mask_muscle * mask_label
            mask_muscle = mask_muscle > 0
            mask_muscle = mask_muscle * 10
            
            muscle_mask_psoas_right =  mask_muscle.reshape(img_array.shape)
            
            mask_undefined = mask_img2 > muscle_psoas_upper 
            mask_undefined = mask_undefined * mask_label
            mask_undefined = mask_img2 < fat_psoas_lower 
            mask_undefined = mask_undefined * mask_label
            mask_undefined = mask_undefined > 0
            mask_undefined = mask_undefined* 50
            
            undefined_mask_psoas_right=  mask_undefined.reshape(img_array.shape)
              
            fat_img = mask_img2 >= fat_psoas_lower
            fat_img = fat_img * mask_label
            fat_img = fat_img > 0
            fat_img = fat_img * 100       
            
            fat_mask_psoas_right = fat_img.reshape(img_array.shape)
                        
           
        if label == 6:
            print(labelname_6)
            mask_label = mask_array == label
            mask_img2 = mask_label * img_array
            mask_img3 = img_array[mask_label]
            
            muscle_img_label = mask_img3 <= muscle_psoas_upper 
            
            undefined_img_label1 = mask_img3 > muscle_psoas_upper 
            undefined_img_label2 = mask_img3 < fat_psoas_lower
            undefined_img_label = undefined_img_label1 * undefined_img_label2
            
            fat_img_label = mask_img3 >= fat_psoas_lower 
            
            total = np.sum(muscle_img_label) + np.sum(undefined_img_label) + np.sum(fat_img_label)
            volume_total = ((sx * sy * sz)* total) / 1000
            muscle_total = np.sum(muscle_img_label)
            undefined_total = np.sum(undefined_img_label)
            fat_total = np.sum(fat_img_label)
            volume_muscle = ((sx * sy * sz)* muscle_total) / 1000
            volume_undefined = ((sx * sy * sz)* undefined_total) / 1000
            volume_fat = ((sx * sy * sz)* fat_total) / 1000
            volume_muscle_psoas_left.append(volume_muscle)
            volume_undefined_psoas_left.append(volume_undefined)
            volume_fat_psoas_left.append(volume_fat)
            muscle_psoas_left.append((muscle_total / total) * 100 )
            undefined_psoas_left.append((undefined_total / total) * 100 )
            fat_psoas_left.append((fat_total / total) * 100)
            volume_total_psoas_left.append(volume_total)
            
            print((muscle_total / total) * 100 )
            print((undefined_total/ total)*100)
            print((fat_total / total) * 100)
        
            mask_muscle = mask_img2 <= muscle_psoas_upper 
            mask_muscle = mask_muscle * mask_label
            mask_muscle = mask_muscle > 0
            mask_muscle = mask_muscle * 10
            
            muscle_mask_psoas_left =  mask_muscle.reshape(img_array.shape)
            
            mask_undefined = mask_img2 > muscle_psoas_upper 
            mask_undefined = mask_undefined * mask_label
            mask_undefined = mask_img2 < fat_psoas_lower 
            mask_undefined = mask_undefined * mask_label
            mask_undefined = mask_undefined > 0
            mask_undefined = mask_undefined* 50
            
            undefined_mask_psoas_left =  mask_undefined.reshape(img_array.shape)
              
            fat_img = mask_img2 >= fat_psoas_lower
            fat_img = fat_img * mask_label
            fat_img = fat_img > 0
            fat_img = fat_img * 100     
            
            fat_mask_psoas_left = fat_img.reshape(img_array.shape)
            
              
    gt_image_data = muscle_mask_multifidus_right + undefined_mask_multifidus_right + fat_mask_multifidus_right + muscle_mask_multifidus_left + undefined_mask_multifidus_left + fat_mask_multifidus_left + muscle_mask_erector_right + undefined_mask_erector_right +fat_mask_erector_right + muscle_mask_erector_left + undefined_mask_erector_left +fat_mask_erector_left + muscle_mask_psoas_right + undefined_mask_psoas_right +fat_mask_psoas_right+ muscle_mask_psoas_left + undefined_mask_psoas_left+ fat_mask_psoas_left
    
    if kmeans:
        gt_file= ID_name_file  + '_Trimodal_Kmeans'+ '_GT.nii.gz'
        gt_img = nib.Nifti1Image(np.rint(gt_image_data), img.affine, img.header)
        gt_img.get_data_dtype() == np.dtype(np.float64)
        gt_img.to_filename(gt_file)
        
    if GMM_on:
        gt_file= ID_name_file  + '_Trimodal_GMM'+ '_GT.nii.gz'
        gt_img = nib.Nifti1Image(np.rint(gt_image_data), img.affine, img.header)
        gt_img.get_data_dtype() == np.dtype(np.float64)
        gt_img.to_filename(gt_file)
        
        
summary_muscle_multifidus_right = [np.mean(muscle_multifidus_right), np.std(muscle_multifidus_right)]
summary_muscle_multifidus_left = [np.mean(muscle_multifidus_left), np.std(muscle_multifidus_left)]

summary_muscle_erector_right = [np.mean(muscle_erector_right), np.std(muscle_erector_right)]
summary_muscle_erector_left = [np.mean(muscle_erector_left), np.std(muscle_erector_left)]

summary_muscle_psoas_right = [np.mean(muscle_psoas_right), np.std(muscle_psoas_right)]
summary_muscle_psoas_left = [np.mean(muscle_psoas_left), np.std(muscle_psoas_left)]

summary_undefined_multifidus_right = [np.mean(undefined_multifidus_right), np.std(undefined_multifidus_right)]
summary_undefined_multifidus_left = [np.mean(undefined_multifidus_left), np.std(undefined_multifidus_left)]

summary_undefined_erector_right = [np.mean(undefined_erector_right), np.std(undefined_erector_right)]
summary_undefined_erector_left = [np.mean(undefined_erector_left), np.std(undefined_erector_left)]

summary_undefined_psoas_right = [np.mean(undefined_psoas_right), np.std(undefined_psoas_right)]
summary_undefined_psoas_left = [np.mean(undefined_psoas_left), np.std(undefined_psoas_left)]

summary_fat_multifidus_right = [np.mean(fat_multifidus_right), np.std(fat_multifidus_right)]
summary_fat_multifidus_left = [np.mean(fat_multifidus_left), np.std(fat_multifidus_left)]

summary_fat_erector_right = [np.mean(fat_erector_right), np.std(fat_erector_right)]
summary_fat_erector_left = [np.mean(fat_erector_left), np.std(fat_erector_left)]

summary_fat_psoas_right = [np.mean(fat_psoas_right), np.std(fat_psoas_right)]
summary_fat_psoas_left = [np.mean(fat_psoas_left), np.std(fat_psoas_left)]

summary_threshold_multifidus_muscle_UD = [np.mean(Threshold_multifidus_muscle_UD),np.std(Threshold_multifidus_muscle_UD)]
summary_threshold_multifidus_UD_fat = [np.mean(Threshold_multifidus_UD_fat ),np.std(Threshold_multifidus_UD_fat )]

summary_threshold_erector_muscle_UD = [np.mean(Threshold_erector_muscle_UD),np.std(Threshold_erector_muscle_UD)]
summary_threshold_erector_UD_fat = [np.mean(Threshold_erector_UD_fat),np.std(Threshold_erector_UD_fat)]

summary_threshold_psoas_muscle_UD = [np.mean(Threshold_psoas_muscle_UD),np.std(Threshold_psoas_muscle_UD)]
summary_threshold_psoas_UD_fat = [np.mean(Threshold_psoas_UD_fat),np.std(Threshold_psoas_UD_fat )]       

if GMM:
    summary_time_multifidus = [np.mean(tik_tok_multifidus), np.std(tik_tok_multifidus)]
    summary_time_erector = [np.mean(tik_tok_erector), np.std(tik_tok_erector)]
    summary_time_psoas = [np.mean(tik_tok_psoas), np.std(tik_tok_psoas)] 

if GMM:
    summary_time_multifidus = [np.mean(tik_tok_multifidus), np.std(tik_tok_multifidus)]
    summary_time_erector = [np.mean(tik_tok_erector), np.std(tik_tok_erector)]
    summary_time_psoas = [np.mean(tik_tok_psoas), np.std(tik_tok_psoas)]  

header = []
header.append('Mean')
header.append('std')   
  
new_list_summary_fraction =[summary_muscle_multifidus_right, summary_undefined_multifidus_right,summary_fat_multifidus_right, summary_muscle_multifidus_left, summary_undefined_multifidus_left,summary_fat_multifidus_left, summary_muscle_erector_right, summary_undefined_erector_right,summary_fat_erector_right, summary_muscle_erector_left, summary_undefined_erector_left,summary_fat_erector_left, 
                            summary_muscle_psoas_right, summary_undefined_psoas_right,summary_fat_psoas_right, summary_muscle_psoas_left, summary_undefined_psoas_left,summary_fat_psoas_left]
new_list_summary_threshold= [header,summary_threshold_multifidus_muscle_UD, summary_threshold_multifidus_UD_fat, summary_threshold_erector_muscle_UD, summary_threshold_erector_UD_fat, summary_threshold_psoas_muscle_UD, summary_threshold_psoas_UD_fat]
new_list_summary_time = [header,summary_time_multifidus, summary_time_erector, summary_time_psoas]
new_list_muscle = [ID_name, muscle_multifidus_right,muscle_multifidus_left,muscle_erector_right,muscle_erector_left,muscle_psoas_right,muscle_psoas_left] 
new_list_fat = [ID_name, fat_multifidus_right,fat_multifidus_left,fat_erector_right,fat_erector_left,fat_psoas_right,fat_psoas_left] 
new_list_undefined= [ID_name, undefined_multifidus_right,undefined_multifidus_left,undefined_erector_right,undefined_erector_left,undefined_psoas_right,undefined_psoas_left]
new_list_total_volume = [ID_name, volume_total_multifidus_right,volume_total_multifidus_left,volume_total_erector_right,volume_total_erector_left,volume_total_psoas_right,volume_total_psoas_left ]
new_list_muscle_volume = [ID_name, volume_muscle_multifidus_right,volume_muscle_multifidus_left,volume_muscle_erector_right,volume_muscle_erector_left,volume_muscle_psoas_right,volume_muscle_psoas_left ]
new_list_undefined_volume = [ID_name, volume_undefined_multifidus_right,volume_undefined_multifidus_left,volume_undefined_erector_right,volume_undefined_erector_left,volume_undefined_psoas_right,volume_undefined_psoas_left ]
new_list_fat_volume = [ID_name, volume_fat_multifidus_right,volume_fat_multifidus_left,volume_fat_erector_right,volume_fat_erector_left,volume_fat_psoas_right,volume_fat_psoas_left ]
new_list_threshold = [Threshold_multifidus_muscle_UD,Threshold_multifidus_UD_fat,Threshold_erector_muscle_UD,Threshold_erector_UD_fat, Threshold_psoas_muscle_UD,Threshold_psoas_UD_fat]
new_list_time = [tik_tok_multifidus, tik_tok_erector, tik_tok_psoas]

df1  = pd.DataFrame(new_list_summary_fraction)
df1  = df1.transpose()
df2  = pd.DataFrame(new_list_summary_threshold)
df2  = df2.transpose()
df3  = pd.DataFrame(new_list_summary_time)
df3  = df3.transpose()
df4  = pd.DataFrame(new_list_muscle)
df4  = df4.transpose()
df5  = pd.DataFrame(new_list_undefined)
df5  = df5.transpose()
df6  = pd.DataFrame(new_list_fat)
df6  = df6.transpose()
df7  = pd.DataFrame(new_list_total_volume)
df7  = df7.transpose()
df8  = pd.DataFrame(new_list_muscle_volume)
df8  = df8.transpose()
df9  = pd.DataFrame(new_list_undefined_volume)
df9  = df9.transpose()
df10 = pd.DataFrame(new_list_fat_volume)
df10 = df10.transpose()
df11 = pd.DataFrame(new_list_threshold)
df11 = df11.transpose()
df12 = pd.DataFrame(new_list_time)
df12 = df12.transpose()

if kmeans: 
    writer = pd.ExcelWriter('Trimodal_Kmeans_final.xlsx', engine='xlsxwriter')
if GMM_on:
    writer = pd.ExcelWriter('Trimodal_GMM_final_test.xlsx', engine='xlsxwriter')
    
df1.to_excel(writer, sheet_name='Summary_fraction', index=False)
df2.to_excel(writer, sheet_name='Summary_threshold', index=False)
df3.to_excel(writer, sheet_name='Summary_time', index=False)
df4.to_excel(writer, sheet_name='Muscle', index=False)
df5.to_excel(writer, sheet_name='Undefined', index=False)
df6.to_excel(writer,sheet_name='Fat',index = False)
df7.to_excel(writer, sheet_name='Total_Volume', index=False)
df8.to_excel(writer,sheet_name = 'Muscle_volume', index = False )
df9.to_excel(writer, sheet_name='Undefined_volume', index=False)
df10.to_excel(writer,sheet_name = 'Fat_volume', index = False)
df11.to_excel(writer, sheet_name='Threshold', index=False)
df12.to_excel(writer,sheet_name = 'Time', index = False)

# Get the xlsxwriter workbook and worksheet objects.
workbook  = writer.book

worksheet_Summary_fraction = writer.sheets['Summary_fraction']
worksheet_Summary_threshold = writer.sheets['Summary_threshold']
worksheet_Summary_time = writer.sheets['Summary_time']
worksheet_Muscle = writer.sheets['Muscle']
worksheet_Undefined = writer.sheets['Undefined']
worksheet_Fat  = writer.sheets['Fat']
worksheet_Total_volume = writer.sheets['Total_Volume']
worksheet_Volume_muscle  = writer.sheets['Muscle_volume']
worksheet_Volume_undefined  = writer.sheets['Undefined_volume']
worksheet_Volume_fat  = writer.sheets['Fat_volume']
worksheet_Threshold  = writer.sheets['Threshold']
worksheet_Time  = writer.sheets['Time']

title_summary = ['Muscle_LMM_right', 'UD_LMM_right', 'FF_LMM_right',  'Muscle_LMM_left','UD_LMM_left','FF_LMM_left', 'Muscle_ES_right','UD_ES_right','FF_ES_right', 
                 'Muscle_ES_left','UD_ES_left','FF_ES_left','Muscle_PS_right','UD_PS_right','FF_PS_right', 'Muscle_PS_left','UD_PS_left','FF_PS_left']
title_time = ['LMM', 'ES', 'PS']
title_threshold = ['LMM_Muscle-UD', 'LMM_UD_Fat','ES_Muscle-UD', 'ES_UD_Fat', 'PS_Muscle-UD', 'PS_UD_Fat']
title = ['LMM right', 'LMM_left', 'ES_right', 'ES_left', 'PS_right', 'PS_left']

row=0
col=1
j = 0
bold = workbook.add_format({'bold': True,'font_color': 'blue'})

for j, t in enumerate(title):
    worksheet_Muscle.write(row, col + j, t, bold)
    worksheet_Undefined.write(row, col + j, t, bold)
    worksheet_Fat.write(row, col + j, t, bold)
    worksheet_Total_volume.write(row, col + j, t, bold)
    worksheet_Volume_muscle.write(row, col + j, t, bold)
    worksheet_Volume_undefined.write(row, col + j, t, bold)
    worksheet_Volume_fat.write(row, col + j, t, bold)
    
row=0
col=1
j = 0
for j, t in enumerate(title_time):
    worksheet_Summary_time.write(row, col + j, t, bold)
    worksheet_Time.write(row, col + j, t, bold)
row=0
col=1
j = 0
for j, t in enumerate(title_threshold):
    worksheet_Summary_threshold.write(row, col + j, t, bold)
    worksheet_Threshold.write(row, col + j, t, bold)
    
row=0
col=1
j = 0
for j, t in enumerate(title_summary):
    worksheet_Summary_fraction.write(row, col + j, t, bold) 

worksheet_Summary_fraction.set_column(0,8,15)
worksheet_Summary_threshold.set_column(0,8,15)
worksheet_Summary_time.set_column(0,8,15)
worksheet_Muscle.set_column(0,8,15)
worksheet_Undefined.set_column(0,8,15)
worksheet_Fat.set_column(0,8,15)
worksheet_Total_volume.set_column(0,8,15)
worksheet_Volume_muscle.set_column(0,8,15)
worksheet_Volume_undefined.set_column(0,8,15)
worksheet_Volume_fat.set_column(0,8,15)
worksheet_Threshold.set_column(0,8,15)
worksheet_Time.set_column(0,8,15)

worksheet_Summary_fraction.set_tab_color('blue')
worksheet_Summary_threshold.set_tab_color('blue')
worksheet_Summary_time.set_tab_color('blue')

writer.save()  
            