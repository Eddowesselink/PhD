import numpy as np 
import os
from glob import glob
import nibabel as nib
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set_style("dark")
import pandas as pd
import seaborn as sns; sns.set()
from sklearn.cluster import KMeans
import scipy.stats as ss
import time 
from sklearn.mixture import GaussianMixture as GMM

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

threshold_multifidus = []
threshold_erector = []
threshold_psoas = []

tik_tok_multifidus = []
tik_tok_erector = []
tik_tok_psoas = []
summary = []
ID = []
ID_name = []

header = []
header.append('Mean')
header.append('std')   

iii = 0
ii = 0
#yy = - 1
for y in glob(os.path.join(data_dir,'Training','*T2.nii.gz')): 
      
    #Load image load 
    img = nib.load(all_images[ii])
    img1 = img.get_fdata()
    sx,sy,sz = img.header['pixdim'][1:4]
    img1 = np.array(img1)
    ID_full = all_images[ii]
    ID_name_file = ID_full[40:46]
    ID.append(ii + 1)
    ID_name.append(ID_name_file)
    print(ID_name_file)
    
    #Load mask
    mask = nib.load(all_masks[ii])
    min_pixdim=np.min(mask.header['pixdim'][1:4])
    mask1 = mask.get_fdata()
    mask1 = np.array(mask1)
    ii =  ii + 1
    
    mask_multifidus1 = mask1 == 1
    mask_multifidus2 = mask1 == 2
    mask_multifidus = mask_multifidus1 + mask_multifidus2
    
    mask_erector1 = mask1 == 3
    mask_erector2 = mask1 == 4
    
    mask_erector = mask_erector1 + mask_erector2
    
    mask_psoas1 = mask1 == 5
    mask_psoas2 = mask1 == 6
    mask_psoas = mask_psoas1 + mask_psoas2 
    
    #get all voxels per muscles combined to calculate GMM
    mask_img_multifidus= img1[mask_multifidus]
    mask_img_erector = img1[mask_erector]
    mask_img_psoas = img1[mask_psoas]

    B = np.reshape(mask_img_multifidus, (-1, 1))
    D = np.reshape(mask_img_erector, (-1, 1))
    C = np.reshape(mask_img_psoas, (-1, 1))    
    
    if kmeans:
        kmeans = KMeans(n_clusters = 2, init = 'k-means++', tol = 0.001, n_init = 50, max_iter = 1000)
        
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
        gmm = GMM(n_components = 2, covariance_type= 'full', init_params = 'kmeans', tol=(0.001), n_init = 50, max_iter = 1000)
        
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
    mask_muscle_multifidus3 =  mask_img_multifidus[mask_muscle_multifidus1]
    mean_mask_muscle_multifidus  = np.mean(mask_muscle_multifidus3)
            
    mask_fat_multifidus1 = labels_multifidus == 1
    mask_fat_multifidus3 =  mask_img_multifidus[mask_fat_multifidus1]
    mean_mask_fat_multifidus = np.mean(mask_fat_multifidus3)

    muscle_fixed_multifidus = mask_muscle_multifidus3
    fat_fixed_multifidus = mask_fat_multifidus3
    
    idxmean_multifidus = np.array([mean_mask_muscle_multifidus,mean_mask_fat_multifidus])
    idxmean_multifidus = ss.rankdata (idxmean_multifidus, method ='min')
    
    if idxmean_multifidus[0] == 2 and idxmean_multifidus[1] == 1: 
        mask_muscle_multifidus3 = fat_fixed_multifidus
        mask_fat_multifidus3 = muscle_fixed_multifidus
        
    muscle_multifidus_upper = max(mask_muscle_multifidus3)
    print('max',muscle_multifidus_upper)
    muscle_multifidus_lower = min(mask_muscle_multifidus3)
    print('min', muscle_multifidus_lower)
    fat_multifidus_upper = max(mask_fat_multifidus3)
    print('max',fat_multifidus_upper)
    fat_multifidus_lower = min(mask_fat_multifidus3)
    print('min', fat_multifidus_lower)
    
    threshold_multifidus.append(muscle_multifidus_upper)
    
    mask_muscle_erector1 = labels_erector == 0
    mask_muscle_erector3 = mask_img_erector[mask_muscle_erector1]
    mean_mask_muscle_erector  = np.mean(mask_muscle_erector3)
            
    mask_fat_erector1 = labels_erector == 1
    mask_fat_erector3=  mask_img_erector[mask_fat_erector1]
    mean_mask_fat_erector= np.mean(mask_fat_erector3)

    muscle_fixed_erector = mask_muscle_erector3
    fat_fixed_erector = mask_fat_erector3
    
    idxmean_erector = np.array([mean_mask_muscle_erector,mean_mask_fat_erector])
    idxmean_erector = ss.rankdata (idxmean_erector, method ='min')
    
    if idxmean_erector[0] == 2 and idxmean_erector[1] == 1: 
        mask_muscle_erector3 = fat_fixed_erector
        mask_fat_erector3 = muscle_fixed_erector
        
    muscle_erector_upper = max(mask_muscle_erector3)
    print('max',muscle_erector_upper)
    muscle_erector_lower = min(mask_muscle_erector3)
    print('min', muscle_erector_lower)
    fat_erector_upper = max(mask_fat_erector3)
    print('max',fat_erector_upper)
    fat_erector_lower = min(mask_fat_erector3)
    print('min', fat_erector_lower)
    
    threshold_erector.append(muscle_erector_upper)
    
    mask_muscle_psoas1 = labels_psoas == 0
    mask_muscle_psoas3 = mask_img_psoas[mask_muscle_psoas1]
    mean_mask_muscle_psoas  = np.mean(mask_muscle_psoas3)
            
    mask_fat_psoas1 = labels_psoas == 1
    mask_fat_psoas3 = mask_img_psoas [mask_fat_psoas1]
    mean_mask_fat_psoas = np.mean(mask_fat_psoas3)
            
    muscle_fixed_psoas = mask_muscle_psoas3
    fat_fixed_psoas = mask_fat_psoas3
    
    idxmean_psoas = np.array([mean_mask_muscle_psoas,mean_mask_fat_psoas])
    idxmean_psoas = ss.rankdata (idxmean_psoas, method ='min')
    
    if idxmean_psoas[0] == 2 and idxmean_psoas[1] == 1: 
        mask_muscle_psoas3 = fat_fixed_psoas
        mask_fat_psoas3 = muscle_fixed_psoas
        
    muscle_psoas_upper = max(mask_muscle_psoas3)
    print('max',muscle_psoas_upper)
    muscle_psoas_lower = min(mask_muscle_psoas3)
    print('min', muscle_psoas_lower)
    fat_psoas_upper = max(mask_fat_psoas3)
    print('max',fat_psoas_upper)
    fat_psoas_lower = min(mask_fat_psoas3)
    print('min', fat_psoas_lower)
    
    threshold_psoas.append(muscle_psoas_upper)

    number_of_labels = mask1.max()
    labelname_1 = 'Multifidus_right'
    labelname_2 = 'Multifidus_left'
    labelname_3 = 'Erector_spinae_right'
    labelname_4 = 'Erector_spinae_left'
    labelname_5 = 'Psoas_right'
    labelname_6 = 'Psoas_left'
        
    for label in range(1,int(number_of_labels)+1):
        
        if label == 1:
            print(labelname_1)
            mask_label = mask1 == label
            mask_img2 = mask_label * img1
            mask_img3 = img1[mask_label]
            
            muscle_img_label = mask_img3 <= muscle_multifidus_upper 
            
            fat_img_label = mask_img3 > muscle_multifidus_upper 
            
            total = np.sum(muscle_img_label) + np.sum(fat_img_label)
            volume = ((sx * sy * sz)* total) / 1000
            muscle_total = np.sum(muscle_img_label)
            fat_total = np.sum(fat_img_label)
            volume_muscle = ((sx * sy * sz)* muscle_total) / 1000
            volume_fat = ((sx * sy * sz)* fat_total) / 1000
            volume_muscle_multifidus_right.append(volume_muscle)
            volume_fat_multifidus_right.append(volume_fat)
            muscle_multifidus_right.append((muscle_total / total) * 100 )
            fat_multifidus_right.append((fat_total / total) * 100)
            volume_total_multifidus_right.append(volume)
            
            print((muscle_total / total) * 100 )
            print((fat_total / total) * 100)
        
            mask_muscle = mask_img2 < muscle_multifidus_upper 
            mask_muscle = mask_muscle * mask_label
            mask_muscle = mask_muscle * 50
            
            muscle_mask_multifidus_right =  mask_muscle.reshape(img1.shape)
            
            fat_img = mask_img2 >= muscle_multifidus_upper 
            fat_img = fat_img * mask_label
            fat_img = fat_img * 100        
            
            fat_mask_multifidus_right = fat_img.reshape(img1.shape)
            
        if label == 2:
            print(labelname_2)
            mask_label = mask1 == label
            mask_img2 = mask_label * img1
            mask_img3 = img1[mask_label]
            
            muscle_img_label = mask_img3 <= muscle_multifidus_upper 
            
            fat_img_label = mask_img3 > muscle_multifidus_upper 
            
            total = np.sum(muscle_img_label) + np.sum(fat_img_label)
            volume = ((sx * sy * sz)* total) / 1000
            muscle_total = np.sum(muscle_img_label)
            fat_total = np.sum(fat_img_label)
            volume_muscle = ((sx * sy * sz)* muscle_total) / 1000
            volume_fat = ((sx * sy * sz)* fat_total) / 1000
            volume_muscle_multifidus_left.append(volume_muscle)
            volume_fat_multifidus_left.append(volume_fat)
            muscle_multifidus_left.append((muscle_total / total) * 100 )
            fat_multifidus_left.append((fat_total / total) * 100)
            volume_total_multifidus_left.append(volume)
            
            print((muscle_total / total) * 100 )
            print((fat_total / total) * 100)
        
            mask_muscle = mask_img2 < muscle_multifidus_upper 
            mask_muscle = mask_muscle * mask_label
            mask_muscle = mask_muscle * 50
            
            muscle_mask_multifidus_left =  mask_muscle.reshape(img1.shape)
            
            fat_img = mask_img2 >= muscle_multifidus_upper 
            fat_img = fat_img * mask_label
            fat_img = fat_img * 100           
            
            fat_mask_multifidus_left = fat_img.reshape(img1.shape)
            
                 
        if label == 3:
            print(labelname_3)
            mask_label = mask1 == label
            mask_img2 = mask_label * img1
            mask_img3 = img1[mask_label]
            
            muscle_img_label = mask_img3 <= muscle_erector_upper 
            
            fat_img_label = mask_img3 > muscle_erector_upper 
            
            total = np.sum(muscle_img_label) + np.sum(fat_img_label)
            volume = ((sx * sy * sz)* total) / 1000
            muscle_total = np.sum(muscle_img_label)
            fat_total = np.sum(fat_img_label)
            volume_muscle = ((sx * sy * sz)* muscle_total) / 1000
            volume_fat = ((sx * sy * sz)* fat_total) / 1000
            volume_muscle_erector_right.append(volume_muscle)
            volume_fat_erector_right.append(volume_fat)
            muscle_erector_right.append((muscle_total / total) * 100 )
            fat_erector_right.append((fat_total / total) * 100)
            volume_total_erector_right.append(volume)
            
            print((muscle_total / total) * 100 )
            print((fat_total / total) * 100)
        
            mask_muscle = mask_img2 < muscle_erector_upper 
            mask_muscle = mask_muscle * mask_label
            mask_muscle = mask_muscle * 50
            
            muscle_mask_erector_right =  mask_muscle.reshape(img1.shape)
            
            fat_img = mask_img2 >= muscle_erector_upper 
            fat_img = fat_img * mask_label
            fat_img = fat_img * 100           
            
            fat_mask_erector_right= fat_img.reshape(img1.shape)
    

        if label == 4:
            print(labelname_4)
            mask_label = mask1 == label
            mask_img2 = mask_label * img1
            mask_img3 = img1[mask_label]
            
            muscle_img_label = mask_img3 <= muscle_erector_upper 
            
            fat_img_label = mask_img3 > muscle_erector_upper 
            
            total = np.sum(muscle_img_label) + np.sum(fat_img_label)
            volume = ((sx * sy * sz)* total) / 1000
            muscle_total = np.sum(muscle_img_label)
            fat_total = np.sum(fat_img_label)
            volume_muscle = ((sx * sy * sz)* muscle_total) / 1000
            volume_fat = ((sx * sy * sz)* fat_total) / 1000
            volume_muscle_erector_left.append(volume_muscle)
            volume_fat_erector_left.append(volume_fat)
            muscle_erector_left.append((muscle_total / total) * 100 )
            fat_erector_left.append((fat_total / total) * 100)
            volume_total_erector_left.append(volume)
            
            print((muscle_total / total) * 100 )
            print((fat_total / total) * 100)
        
            mask_muscle = mask_img2 < muscle_erector_upper 
            mask_muscle = mask_muscle * mask_label
            mask_muscle = mask_muscle * 50
            
            muscle_mask_erector_left =  mask_muscle.reshape(img1.shape)
            
            fat_img = mask_img2 >= muscle_erector_upper 
            fat_img = fat_img * mask_label
            fat_img = fat_img * 100           
            
            fat_mask_erector_left = fat_img.reshape(img1.shape)
            
        if label == 5:
            print(labelname_5)
            mask_label = mask1 == label
            mask_img2 = mask_label * img1
            mask_img3 = img1[mask_label]
            
            muscle_img_label = mask_img3 <= muscle_psoas_upper 
            
            fat_img_label = mask_img3 > muscle_psoas_upper 
            
            total = np.sum(muscle_img_label) + np.sum(fat_img_label)
            volume = ((sx * sy * sz)* total) / 1000
            muscle_total = np.sum(muscle_img_label)
            fat_total = np.sum(fat_img_label)
            volume_muscle = ((sx * sy * sz)* muscle_total) / 1000
            volume_fat = ((sx * sy * sz)* fat_total) / 1000
            volume_muscle_psoas_right.append(volume_muscle)
            volume_fat_psoas_right.append(volume_fat)
            muscle_psoas_right.append((muscle_total / total) * 100 )
            fat_psoas_right.append((fat_total / total) * 100)
            volume_total_psoas_right.append(volume)
            
            print((muscle_total / total) * 100 )
            print((fat_total / total) * 100)
        
            mask_muscle = mask_img2 < muscle_psoas_upper 
            mask_muscle = mask_muscle * mask_label
            mask_muscle = mask_muscle * 50
            
            muscle_mask_psoas_right =  mask_muscle.reshape(img1.shape)
            
            fat_img = mask_img2 >= muscle_psoas_upper 
            fat_img = fat_img * mask_label
            fat_img = fat_img * 100          
            
            fat_mask_psoas_right = fat_img.reshape(img1.shape)
                        
        if label == 6:
            print(labelname_6)
            mask_label = mask1 == label
            mask_img2 = mask_label * img1
            mask_img3 = img1[mask_label]
            
            muscle_img_label = mask_img3 <= muscle_psoas_upper
            
            fat_img_label = mask_img3 > muscle_psoas_upper
            
            total = np.sum(muscle_img_label) + np.sum(fat_img_label)
            volume = ((sx * sy * sz)* total) / 1000
            muscle_total = np.sum(muscle_img_label)
            fat_total = np.sum(fat_img_label)
            volume_muscle = ((sx * sy * sz)* muscle_total) / 1000
            volume_fat = ((sx * sy * sz)* fat_total) / 1000
            volume_muscle_psoas_left.append(volume_muscle)
            volume_fat_psoas_left.append(volume_fat)
            muscle_psoas_left.append((muscle_total / total) * 100 )
            fat_psoas_left.append((fat_total / total) * 100)
            volume_total_psoas_left.append(volume)
            
            print((muscle_total / total) * 100 )
            print((fat_total / total) * 100)
        
            mask_muscle = mask_img2 < muscle_psoas_upper
            mask_muscle = mask_muscle * mask_label
            mask_muscle = mask_muscle * 50
            
            muscle_mask_psoas_left =  mask_muscle.reshape(img1.shape)
            
            fat_img = mask_img2 >= muscle_psoas_upper
            fat_img = fat_img * mask_label
            fat_img = fat_img * 100          
            
            fat_mask_psoas_left= fat_img.reshape(img1.shape)
            
              
    gt_image_data = muscle_mask_multifidus_right + fat_mask_multifidus_right + muscle_mask_multifidus_left + fat_mask_multifidus_left + muscle_mask_erector_right + fat_mask_erector_right + muscle_mask_erector_left + fat_mask_erector_left + muscle_mask_psoas_right + fat_mask_psoas_right+ muscle_mask_psoas_left + fat_mask_psoas_left
    
    if kmeans:
        
        gt_file= ID_name_file + '_Bimodal_Kmeans' + '_GT.nii.gz'
        gt_img = nib.Nifti1Image(np.rint(gt_image_data), img.affine, img.header)
        gt_img.get_data_dtype() == np.dtype(np.float64)
        gt_img.to_filename(gt_file)
    
    if GMM_on:
        
        gt_file= ID_name_file + '_Bimodal_GMM' + '_GT.nii.gz'
        gt_img = nib.Nifti1Image(np.rint(gt_image_data), img.affine, img.header)
        gt_img.get_data_dtype() == np.dtype(np.float64)
        gt_img.to_filename(gt_file)    

    
summary_muscle_multifidus_right = [np.mean(muscle_multifidus_right), np.std(muscle_multifidus_right)]
summary_muscle_multifidus_left = [np.mean(muscle_multifidus_left), np.std(muscle_multifidus_left)]

summary_fat_multifidus_right = [np.mean(fat_multifidus_right), np.std(fat_multifidus_right)]
summary_fat_multifidus_left = [np.mean(fat_multifidus_left), np.std(fat_multifidus_left)]

summary_muscle_erector_right = [np.mean(muscle_erector_right), np.std(muscle_erector_right)]
summary_muscle_erector_left = [np.mean(muscle_erector_left), np.std(muscle_erector_left)]

summary_fat_erector_right = [np.mean(fat_erector_right), np.std(fat_erector_right)]
summary_fat_erector_left = [np.mean(fat_erector_left), np.std(fat_erector_left)]

summary_muscle_psoas_right = [np.mean(muscle_psoas_right), np.std(muscle_psoas_right)]
summary_muscle_psoas_left = [np.mean(muscle_psoas_left), np.std(muscle_psoas_left)]

summary_fat_psoas_right = [np.mean(fat_psoas_right), np.std(fat_psoas_right)]
summary_fat_psoas_left = [np.mean(fat_psoas_left), np.std(fat_psoas_left)]

summary_threshold_multifidus = [np.mean(threshold_multifidus),np.std(threshold_multifidus)]
summary_threshold_erector = [np.mean(threshold_erector),np.std(threshold_erector)]
summary_threshold_psoas = [np.mean(threshold_psoas),np.std(threshold_psoas)]

if kmeans:
    summary_time_multifidus = [np.mean(tik_tok_multifidus), np.std(tik_tok_multifidus)]
    summary_time_erector = [np.mean(tik_tok_erector), np.std(tik_tok_erector)]
    summary_time_psoas = [np.mean(tik_tok_psoas), np.std(tik_tok_psoas)] 

if GMM_on:
    summary_time_multifidus = [np.mean(tik_tok_multifidus), np.std(tik_tok_multifidus)]
    summary_time_erector = [np.mean(tik_tok_erector), np.std(tik_tok_erector)]
    summary_time_psoas = [np.mean(tik_tok_psoas), np.std(tik_tok_psoas)] 
    
new_list_summary =[header,summary_muscle_multifidus_right, summary_fat_multifidus_right, summary_muscle_multifidus_left, summary_fat_multifidus_left, summary_muscle_erector_right,summary_fat_erector_right, 
                   summary_muscle_erector_left, summary_fat_erector_left,summary_muscle_psoas_right,summary_fat_psoas_right, summary_muscle_psoas_left,summary_fat_psoas_left]
new_list_summary_threshold = [header,summary_threshold_multifidus,summary_threshold_erector,summary_threshold_psoas]
new_list_summary_time = [header,summary_time_multifidus, summary_time_erector, summary_time_psoas]
new_list_muscle = [ID_name, muscle_multifidus_right,muscle_multifidus_left,muscle_erector_right,muscle_erector_left,muscle_psoas_right,muscle_psoas_left] 
new_list_fat = [ID_name, fat_multifidus_right,fat_multifidus_left,fat_erector_right,fat_erector_left,fat_psoas_right,fat_psoas_left] 
new_list_volume = [ID_name, volume_total_multifidus_right,volume_total_multifidus_left,volume_total_erector_right,volume_total_erector_left,volume_total_psoas_right,volume_total_psoas_left ]
new_list_muscle_volume = [ID_name, volume_muscle_multifidus_right,volume_muscle_multifidus_left,volume_muscle_erector_right,volume_muscle_erector_left,volume_muscle_psoas_right,volume_muscle_psoas_left ]
new_list_fat_volume = [ volume_fat_multifidus_right,volume_fat_multifidus_left,volume_fat_erector_right,volume_fat_erector_left,volume_fat_psoas_right,volume_fat_psoas_left ]
new_list_threshold = [threshold_multifidus,threshold_erector,threshold_psoas]
new_list_time = [tik_tok_multifidus, tik_tok_erector, tik_tok_psoas]

df1 = pd.DataFrame(new_list_summary)
df1 = df1.transpose()
df2 = pd.DataFrame(new_list_summary_threshold)
df2 = df2.transpose()
df3 = pd.DataFrame(new_list_summary_time)
df3 = df3.transpose()
df4 = pd.DataFrame(new_list_muscle)
df4 = df4.transpose()
df5 = pd.DataFrame(new_list_fat)
df5 = df5.transpose()
df6 = pd.DataFrame(new_list_volume)
df6 = df6.transpose()
df7 = pd.DataFrame(new_list_muscle_volume)
df7 = df7.transpose()
df8 = pd.DataFrame(new_list_fat_volume)
df8 = df8.transpose()
df9 = pd.DataFrame(new_list_threshold)
df9 = df9.transpose()
df10 = pd.DataFrame(new_list_time)
df10 = df10.transpose()

if kmeans:
    writer = pd.ExcelWriter('Bimodal_Kmeans_final_2.xlsx', engine='xlsxwriter')
if GMM_on:
    writer = pd.ExcelWriter('Bimodal_GMM_final_2.xlsx', engine='xlsxwriter')

df1.to_excel(writer, sheet_name='Summary', index=False)   
df2.to_excel(writer, sheet_name='Summary_threshold', index=False)   
df3.to_excel(writer, sheet_name='Summary_time', index=False)   
df4.to_excel(writer, sheet_name='Muscle', index=False)
df5.to_excel(writer, sheet_name='Fatty infiltration', index=False)
df6.to_excel(writer, sheet_name='Total Volume', index=False)
df7.to_excel(writer,sheet_name = 'Volume Muscle', index = False )
df8.to_excel(writer,sheet_name = 'Volume Fat', index = False )
df9.to_excel(writer,sheet_name = 'Threshold', index = False )
df10.to_excel(writer,sheet_name = 'Time', index = False )

# Get the xlsxwriter workbook and worksheet objects.
workbook  = writer.book

worksheet_Summary = writer.sheets['Summary']
worksheet_Summary_threshold = writer.sheets['Summary_threshold']
worksheet_Summary_time = writer.sheets['Summary_time']
worksheet_Muscle = writer.sheets['Muscle']
worksheet_Fat  = writer.sheets['Fatty infiltration']
worksheet_Total_volume = writer.sheets['Total Volume']
worksheet_Volume_muscle  = writer.sheets['Volume Muscle']
worksheet_Volume_fat  = writer.sheets['Volume Fat']
worksheet_Threshold  = writer.sheets['Threshold']
worksheet_Time  = writer.sheets['Time']


title_summary = ['Muscle_LMM_right', 'FF_LMM_right', 'Muscle_LMM_left', 'FF_LMM_left', 'Muscle_ES_right', 'FF_ES_right', 'Muscle_ES_left', 'FF_ES_left', 
                 'Muscle_PS_right', 'FF_PS_right', 'Muscle_PS_left', 'FF_PS_left']
title_threshold_time = ['LMM', 'ES', 'PS']
title = ['LMM right', 'LMM_left', 'ES_right', 'ES_left', 'PS_right', 'PS_left']
row=0
col=1
j = 0
bold = workbook.add_format({'bold': True,'font_color': 'blue'})

for j, t in enumerate(title):
    worksheet_Muscle.write(row, col + j, t, bold)
    worksheet_Fat.write(row, col + j, t, bold)
    worksheet_Total_volume.write(row, col + j, t, bold)
    worksheet_Volume_muscle.write(row, col + j, t, bold)
    worksheet_Volume_fat.write(row, col + j, t, bold)
    
row=0
col=0
j = 0
for j, t in enumerate(title_threshold_time):
    worksheet_Threshold.write(row, col + j, t, bold)
    worksheet_Time.write(row, col + j, t, bold)
    
row=0
col=1
j = 0
for j, t in enumerate(title_threshold_time):
    worksheet_Summary_threshold.write(row, col + j, t, bold)
    worksheet_Summary_time.write(row, col + j, t, bold)

row=0
col=1
j = 0
for j, t in enumerate(title_summary):
    worksheet_Summary.write(row, col + j, t, bold) 


worksheet_Summary.set_column(0,12,18)
worksheet_Summary_threshold.set_column(0,8,15)
worksheet_Summary_time.set_column(0,8,15)
worksheet_Muscle.set_column(0,8,15)
worksheet_Fat.set_column(0,8,15)
worksheet_Total_volume.set_column(0,8,15)
worksheet_Volume_muscle.set_column(0,8,15)
worksheet_Volume_fat.set_column(0,8,15)
worksheet_Threshold.set_column(0, 8,15)
worksheet_Time.set_column(0, 8,15)

worksheet_Summary.set_tab_color('blue')
worksheet_Summary_threshold.set_tab_color('blue')
worksheet_Summary_time.set_tab_color('blue')

writer.save()

            