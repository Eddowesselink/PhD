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

def Thresholding(mask_img_multifidus,mask_img_erector,mask_img_psoas, ncomponents, algorithm):
        if algorithm == 'kmeans':
            kmeans = KMeans(n_clusters = ncomponents, init = 'k-means++', tol = 0.001, n_init = 50, max_iter = 1000)
            labels_multifidus = kmeans.fit(mask_img_multifidus).predict(mask_img_multifidus)
            labels_erector = kmeans.fit(mask_img_erector).predict(mask_img_erector)
            labels_psoas = kmeans.fit(mask_img_psoas).predict(mask_img_psoas)
        if algorithm == 'GMM':
            gmm = GMM(n_components = ncomponents, covariance_type= 'full', init_params = 'kmeans', tol=(0.001), n_init = 50, max_iter = 1000)
            labels_multifidus = gmm.fit(mask_img_multifidus).predict(mask_img_multifidus)
            labels_erector = gmm.fit(mask_img_erector).predict(mask_img_erector)
            labels_psoas = gmm.fit(mask_img_psoas).predict(mask_img_psoas)
        else:
            raise ValueError("Invalid clustering type. Supported types are 'kmeans' and 'GMM'")
        return labels_multifidus, labels_erector, labels_psoas
    
