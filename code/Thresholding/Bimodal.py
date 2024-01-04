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
    ID_name_file = img[40:46]
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

    #Run Kmeans or GMM
    if kmeans_activate:
        kmeans = KMeans(n_clusters = 2, init = 'k-means++', tol = 0.001, n_init = 50, max_iter = 1000)

        start = time.time()
        labels_multifidus = kmeans.fit(mask_img_multifidus).predict(mask_img_multifidus)
        stop = time.time()
        tik_tok_multifidus.append(stop - start)
        
        start = time.time()
        labels_erector = kmeans.fit(mask_img_erector).predict(mask_img_erector)
        stop = time.time()
        tik_tok_erector.append(stop - start)
        
        start = time.time()
        labels_psoas = kmeans.fit(mask_img_psoas).predict(mask_img_psoas)
        stop = time.time()
        tik_tok_psoas.append(stop - start)
        
    if GMM_activate:
        gmm = GMM(n_components = 2, covariance_type= 'full', init_params = 'kmeans', tol=(0.001), n_init = 50, max_iter = 1000)
        
        start = time.time()
        labels_multifidus = gmm.fit(mask_img_multifidus).predict(mask_img_multifidus)
        stop = time.time()
        tik_tok_multifidus.append(stop - start)
        
        start = time.time()
        labels_erector = gmm.fit(mask_img_erector).predict(mask_img_erector)
        stop = time.time()
        tik_tok_erector.append(stop - start)
        
        start = time.time()
        labels_psoas = gmm.fit(mask_img_psoas).predict(mask_img_psoas)
        stop = time.time()
        tik_tok_psoas.append(stop - start)

    # Create a new mask for only the muscle values
    mask_muscle_multifidus = labels_multifidus == 0
    mask_muscle_multifidus =  mask_img_multifidus[mask_muscle_multifidus]

    # Create a new mask for only the fat values    
    mask_fat_multifidus = labels_multifidus == 1
    mask_fat_multifidus =  mask_img_multifidus[mask_fat_multifidus]

    #Get the mean of the arrays.
    mean_muscle_multifidus = np.mean(mask_muscle_multifidus)
    mean_fat_multifidus = np.mean(mask_fat_multifidus)
    idxmean_multifidus = np.array((mean_muscle_multifidus, mean_fat_multifidus))
    idxmean_multifidus = ss.rankdata (idxmean_multifidus, method ='min')

    #create a fixed value
    fixed_muscle_multifidus = mask_muscle_multifidus
    fixed_fat_multifidus = mask_fat_multifidus

    # Here check if the mean of the muscle array is lower than the fat (this should be), otherwise swap
    # This controlling mechanism is important because the clustering techniques are not labelling clusters consistently 
    if idxmean_multifidus[0] == 2 and idxmean_multifidus[1] == 1: 
        mask_muscle_multifidus = fixed_fat_multifidus
        mask_fat_multifidus = fixed_muscle_multifidus
        
    muscle_multifidus_upper = max(mask_muscle_multifidus)
    threshold_multifidus.append(muscle_multifidus_upper)
    
    ##Erector spinae

    # Create a new mask for only the muscle values
    mask_muscle_erector = labels_erector == 0
    mask_muscle_erector =  mask_img_erector[mask_muscle_erector]

    # Create a new mask for only the fat values    
    mask_fat_erector = labels_erector == 1
    mask_fat_erector =  mask_img_erector[mask_fat_erector]

    #Get the mean of the arrays.
    mean_muscle_erector = np.mean(mask_muscle_erector)
    mean_fat_erector = np.mean(mask_fat_erector)
    idxmean_erector = np.array((mean_muscle_erector, mean_fat_erector))
    idxmean_erector = ss.rankdata (idxmean_erector, method ='min')

    #create a fixed value
    fixed_muscle_erector = mask_muscle_erector
    fixed_fat_erector = mask_fat_erector

    # Here check if the mean of the muscle array is lower than the fat (this should be), otherwise swap
    # This controlling mechanism is important because the clustering techniques are not labelling clusters consistently 
    if idxmean_erector[0] == 2 and idxmean_erector[1] == 1: 
        mask_muscle_erector = fixed_fat_erector
        mask_fat_erector = fixed_muscle_erector
        
    muscle_erector_upper = max(mask_muscle_erector)
    threshold_erector.append(muscle_erector_upper)
    
    ## Psoas Major
    # Create a new mask for only the muscle values
    mask_muscle_psoas = labels_psoas == 0
    mask_muscle_psoas =  mask_img_psoas[mask_muscle_psoas]

    # Create a new mask for only the fat values    
    mask_fat_psoas = labels_psoas == 1
    mask_fat_psoas =  mask_img_psoas[mask_fat_psoas]

    #Get the mean of the arrays.
    mean_muscle_psoas = np.mean(mask_muscle_psoas)
    mean_fat_psoas = np.mean(mask_fat_psoas)
    idxmean_psoas = np.array((mean_muscle_psoas, mean_fat_psoas))
    idxmean_psoas = ss.rankdata (idxmean_psoas, method ='min')

    #create a fixed value
    fixed_muscle_psoas = mask_muscle_psoas
    fixed_fat_psoas = mask_fat_psoas

    # Here check if the mean of the muscle array is lower than the fat (this should be), otherwise swap
    # This controlling mechanism is important because the clustering techniques are not labelling clusters consistently 
    if idxmean_psoas[0] == 2 and idxmean_psoas[1] == 1: 
        mask_muscle_psoas = fixed_fat_psoas
        mask_fat_psoas = fixed_muscle_psoas
        
    muscle_psoas_upper = max(mask_muscle_psoas)
    threshold_psoas.append(muscle_psoas_upper)
    
    #grab number of labels for iterator
    number_of_labels = mask_array.max()

    for label in range(1,int(number_of_labels)+1):

        if label == 1:
            #Get only the voxels for ROI (i.e, excluding background)
            mask_label = mask_array == label
            mask_img = img_array[mask_label]

            #Specify voxels for muscle and fat using the threshold
            muscle_img_label = mask_img <= muscle_multifidus_upper 
            fat_img_label = mask_img > muscle_multifidus_upper 
        
            #Calculate fraction and append
            muscle = np.sum(muscle_img_label)
            fat = np.sum(fat_img_label)
            total = muscle + fat
            muscle_multifidus_right.append((muscle / total) * 100 )
            fat_multifidus_right.append((fat / total) * 100)
            
            #Calculate volume in ml
            volume_muscle_multifidus_right.append(((sx * sy * sz)* muscle) / 1000)
            volume_fat_multifidus_right.append(((sx * sy * sz)* fat) / 1000)
            volume_total_multifidus_right.append(((sx * sy * sz)* total) / 1000)
        
            ##Map the image back and give the value a colour by asigning a value
            mask_img = mask_label * img_array
            muscle_img = mask_img < muscle_multifidus_upper 
            muscle_img = muscle_img  * mask_label
            muscle_img = muscle_img  * 50
            muscle_img_multifidus_right =  muscle_img .reshape(img_array.shape)
            
            fat_img = mask_img >= muscle_multifidus_upper 
            fat_img = fat_img * mask_label
            fat_img = fat_img * 100        

            fat_img_multifidus_right = fat_img.reshape(img_array.shape)
        
        if label == 2:
            #Get only the voxels for ROI (i.e, excluding background)
            mask_label = mask_array == label
            mask_img = img_array[mask_label]

            #Specify voxels for muscle and fat using the threshold
            muscle_img_label = mask_img <= muscle_multifidus_upper 
            fat_img_label = mask_img > muscle_multifidus_upper 
        
            #Calculate fraction and append
            muscle = np.sum(muscle_img_label)
            fat = np.sum(fat_img_label)
            total = muscle + fat
            muscle_multifidus_left.append((muscle / total) * 100 )
            fat_multifidus_left.append((fat / total) * 100)
            
            #Calculate volume in ml
            volume_muscle_multifidus_left.append(((sx * sy * sz)* muscle) / 1000)
            volume_fat_multifidus_left.append(((sx * sy * sz)* fat) / 1000)
            volume_total_multifidus_left.append(((sx * sy * sz)* total) / 1000)
        
            ##Map the image back and give the value a colour by asigning a value
            mask_img = mask_label * img_array
            muscle_img = mask_img < muscle_multifidus_upper 
            muscle_img = muscle_img  * mask_label
            muscle_img = muscle_img  * 50
            muscle_img_multifidus_left =  muscle_img .reshape(img_array.shape)
            
            fat_img = mask_img >= muscle_multifidus_upper 
            fat_img = fat_img * mask_label
            fat_img = fat_img * 100        

            fat_img_multifidus_left = fat_img.reshape(img_array.shape)

        if label == 3:
            #Get only the voxels for ROI (i.e, excluding background)
            mask_label = mask_array == label
            mask_img = img_array[mask_label]

            #Specify voxels for muscle and fat using the threshold
            muscle_img_label = mask_img <= muscle_erector_upper 
            fat_img_label = mask_img > muscle_erector_upper 
        
            #Calculate fraction and append
            muscle = np.sum(muscle_img_label)
            fat = np.sum(fat_img_label)
            total = muscle + fat
            muscle_erector_right.append((muscle / total) * 100 )
            fat_erector_right.append((fat / total) * 100)
            
            #Calculate volume in ml
            volume_muscle_erector_right.append(((sx * sy * sz)* muscle) / 1000)
            volume_fat_erector_right.append(((sx * sy * sz)* fat) / 1000)
            volume_total_erector_right.append(((sx * sy * sz)* total) / 1000)
        
            ##Map the image back and give the value a colour by asigning a value
            mask_img = mask_label * img_array
            muscle_img = mask_img < muscle_erector_upper 
            muscle_img = muscle_img  * mask_label
            muscle_img = muscle_img  * 50
            muscle_img_erector_right =  muscle_img .reshape(img_array.shape)
            
            fat_img = mask_img >= muscle_erector_upper 
            fat_img = fat_img * mask_label
            fat_img = fat_img * 100        

            fat_img_erector_right = fat_img.reshape(img_array.shape)
        
        if label == 4:
            #Get only the voxels for ROI (i.e, excluding background)
            mask_label = mask_array == label
            mask_img = img_array[mask_label]

            #Specify voxels for muscle and fat using the threshold
            muscle_img_label = mask_img <= muscle_erector_upper 
            fat_img_label = mask_img > muscle_erector_upper 
        
            #Calculate fraction and append
            muscle = np.sum(muscle_img_label)
            fat = np.sum(fat_img_label)
            total = muscle + fat
            muscle_erector_left.append((muscle / total) * 100 )
            fat_erector_left.append((fat / total) * 100)
            
            #Calculate volume in ml
            volume_muscle_erector_left.append(((sx * sy * sz)* muscle) / 1000)
            volume_fat_erector_left.append(((sx * sy * sz)* fat) / 1000)
            volume_total_erector_left.append(((sx * sy * sz)* total) / 1000)
        
            ##Map the image back and give the value a colour by asigning a value
            mask_img = mask_label * img_array
            muscle_img = mask_img < muscle_erector_upper 
            muscle_img = muscle_img  * mask_label
            muscle_img = muscle_img  * 50
            muscle_img_erector_left =  muscle_img .reshape(img_array.shape)
            
            fat_img = mask_img >= muscle_erector_upper 
            fat_img = fat_img * mask_label
            fat_img = fat_img * 100        

            fat_img_erector_left = fat_img.reshape(img_array.shape)  

        if label == 5:
            #Get only the voxels for ROI (i.e, excluding background)
            mask_label = mask_array == label
            mask_img = img_array[mask_label]

            #Specify voxels for muscle and fat using the threshold
            muscle_img_label = mask_img <= muscle_psoas_upper 
            fat_img_label = mask_img > muscle_psoas_upper 
        
            #Calculate fraction and append
            muscle = np.sum(muscle_img_label)
            fat = np.sum(fat_img_label)
            total = muscle + fat
            muscle_psoas_right.append((muscle / total) * 100 )
            fat_psoas_right.append((fat / total) * 100)
            
            #Calculate volume in ml
            volume_muscle_psoas_right.append(((sx * sy * sz)* muscle) / 1000)
            volume_fat_psoas_right.append(((sx * sy * sz)* fat) / 1000)
            volume_total_psoas_right.append(((sx * sy * sz)* total) / 1000)
        
            ##Map the image back and give the value a colour by asigning a value
            mask_img = mask_label * img_array
            muscle_img = mask_img < muscle_psoas_upper 
            muscle_img = muscle_img  * mask_label
            muscle_img = muscle_img  * 50
            muscle_img_psoas_right =  muscle_img .reshape(img_array.shape)
            
            fat_img = mask_img >= muscle_psoas_upper 
            fat_img = fat_img * mask_label
            fat_img = fat_img * 100        

            fat_img_psoas_right = fat_img.reshape(img_array.shape)
        
        if label == 6:
            #Get only the voxels for ROI (i.e, excluding background)
            mask_label = mask_array == label
            mask_img = img_array[mask_label]

            #Specify voxels for muscle and fat using the threshold
            muscle_img_label = mask_img <= muscle_psoas_upper 
            fat_img_label = mask_img > muscle_psoas_upper 
        
            #Calculate fraction and append
            muscle = np.sum(muscle_img_label)
            fat = np.sum(fat_img_label)
            total = muscle + fat
            muscle_psoas_left.append((muscle / total) * 100 )
            fat_psoas_left.append((fat / total) * 100)
            
            #Calculate volume in ml
            volume_muscle_psoas_left.append(((sx * sy * sz)* muscle) / 1000)
            volume_fat_psoas_left.append(((sx * sy * sz)* fat) / 1000)
            volume_total_psoas_left.append(((sx * sy * sz)* total) / 1000)
        
            ##Map the image back and give the value a colour by asigning a value
            mask_img = mask_label * img_array
            muscle_img = mask_img < muscle_psoas_upper 
            muscle_img = muscle_img  * mask_label
            muscle_img = muscle_img  * 50
            muscle_img_psoas_left =  muscle_img .reshape(img_array.shape)
            
            fat_img = mask_img >= muscle_psoas_upper 
            fat_img = fat_img * mask_label
            fat_img = fat_img * 100        

            fat_img_psoas_left = fat_img.reshape(img_array.shape)  

    gt_image_data = muscle_img_multifidus_right + fat_img_multifidus_right + muscle_img_multifidus_left + fat_img_multifidus_left + muscle_img_erector_right + fat_img_erector_right + muscle_img_erector_left + fat_img_erector_left + muscle_img_psoas_right + fat_img_psoas_right+ muscle_img_psoas_left + fat_img_psoas_left
    
    if kmeans_activate:
        gt_file= ID_name_file + '_Bimodal_Kmeans' + '_GT.nii.gz'
        gt_img = nib.Nifti1Image(np.rint(gt_image_data), img.affine, img.header)
        gt_img.get_data_dtype() == np.dtype(np.float64)
        gt_img.to_filename(gt_file)
    
    if GMM_activate:
        gt_file= ID_name_file + '_Bimodal_GMM' + '_GT.nii.gz'
        gt_img = nib.Nifti1Image(np.rint(gt_image_data), img.affine, img.header)
        gt_img.get_data_dtype() == np.dtype(np.float64)
        gt_img.to_filename(gt_file)    

#create summary values for mean values for fractions, thresholds and time)    
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

summary_time_multifidus = [np.mean(tik_tok_multifidus), np.std(tik_tok_multifidus)]
summary_time_erector = [np.mean(tik_tok_erector), np.std(tik_tok_erector)]
summary_time_psoas = [np.mean(tik_tok_psoas), np.std(tik_tok_psoas)]

#Create lists for excel setup    
list_summary_fraction =[header,summary_muscle_multifidus_right, summary_fat_multifidus_right, summary_muscle_multifidus_left, summary_fat_multifidus_left, summary_muscle_erector_right,summary_fat_erector_right, 
                   summary_muscle_erector_left, summary_fat_erector_left,summary_muscle_psoas_right,summary_fat_psoas_right, summary_muscle_psoas_left,summary_fat_psoas_left]
list_summary_threshold = [header,summary_threshold_multifidus,summary_threshold_erector,summary_threshold_psoas]
list_summary_time = [header,summary_time_multifidus, summary_time_erector, summary_time_psoas]
list_muscle = [ID_name, muscle_multifidus_right,muscle_multifidus_left,muscle_erector_right,muscle_erector_left,muscle_psoas_right,muscle_psoas_left] 
list_fat = [ID_name, fat_multifidus_right,fat_multifidus_left,fat_erector_right,fat_erector_left,fat_psoas_right,fat_psoas_left] 
list_ROI_volume = [ID_name, volume_total_multifidus_right,volume_total_multifidus_left,volume_total_erector_right,volume_total_erector_left,volume_total_psoas_right,volume_total_psoas_left ]
muscle_volume = [ID_name, volume_muscle_multifidus_right,volume_muscle_multifidus_left,volume_muscle_erector_right,volume_muscle_erector_left,volume_muscle_psoas_right,volume_muscle_psoas_left ]
list_fat_volume = [ volume_fat_multifidus_right,volume_fat_multifidus_left,volume_fat_erector_right,volume_fat_erector_left,volume_fat_psoas_right,volume_fat_psoas_left ]
list_threshold = [threshold_multifidus,threshold_erector,threshold_psoas]
list_time = [tik_tok_multifidus, tik_tok_erector, tik_tok_psoas]

df1 = pd.DataFrame(list_summary_fraction)
df1 = df1.transpose()
df2 = pd.DataFrame(list_summary_threshold)
df2 = df2.transpose()
df3 = pd.DataFrame(list_summary_time)
df3 = df3.transpose()
df4 = pd.DataFrame(list_muscle)
df4 = df4.transpose()
df5 = pd.DataFrame(list_fat)
df5 = df5.transpose()
df6 = pd.DataFrame(list_volume)
df6 = df6.transpose()
df7 = pd.DataFrame(list_muscle_volume)
df7 = df7.transpose()
df8 = pd.DataFrame(list_fat_volume)
df8 = df8.transpose()
df9 = pd.DataFrame(list_threshold)
df9 = df9.transpose()
df10 = pd.DataFrame(list_time)
df10 = df10.transpose()

#Create a writer and name to either Kmeans or GMM
if kmeans_activate:
    writer = pd.ExcelWriter('Bimodal_Kmeans_final_2.xlsx', engine='xlsxwriter')
if GMM_activate:
    writer = pd.ExcelWriter('Bimodal_GMM_final_2.xlsx', engine='xlsxwriter')

#set Panda dataframes to excel
df1.to_excel(writer, sheet_name='Summary fraction', index=False)   
df2.to_excel(writer, sheet_name='Summary threshold', index=False)   
df3.to_excel(writer, sheet_name='Summary time', index=False)   
df4.to_excel(writer, sheet_name='Muscle fracction', index=False)
df5.to_excel(writer, sheet_name='Fat fraction', index=False)
df6.to_excel(writer, sheet_name='Total ROI Volume', index=False)
df7.to_excel(writer,sheet_name = 'Muscle volume', index = False )
df8.to_excel(writer,sheet_name = 'Fat volume', index = False )
df9.to_excel(writer,sheet_name = 'Threshold', index = False )
df10.to_excel(writer,sheet_name = 'Time', index = False )

# Get the xlsxwriter workbook and worksheet objects.
workbook  = writer.book
worksheet_Summary_fraction = writer.sheets['Summary fraction']
worksheet_Summary_threshold = writer.sheets['Summary threshold']
worksheet_Summary_time = writer.sheets['Summary time']
worksheet_Muscle = writer.sheets['Muscle fraction']
worksheet_Fat  = writer.sheets['Fat fraction']
worksheet_Total_volume = writer.sheets['Total ROI Volume']
worksheet_Volume_muscle  = writer.sheets['Muscle volume']
worksheet_Volume_fat = writer.sheets['Fat volume']
worksheet_Threshold  = writer.sheets['Threshold']
worksheet_Time  = writer.sheets['Time']


title_summary = ['Muscle_LMM_right', 'Fat_LMM_right', 'Muscle_LMM_left', 'Fat_LMM_left', 'Muscle_ES_right', 'Fat_ES_right', 'Muscle_ES_left', 'Fat_ES_left', 
                 'Muscle_PS_right', 'Fat_PS_right', 'Muscle_PS_left', 'Fat_PS_left']
title_threshold_time = ['LMM', 'ES', 'PS']
title = ['LMM right', 'LMM left', 'ES right', 'ES left', 'PS right', 'PS left']
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

worksheet_Summary_fraction.set_column(0,12,18)
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

            