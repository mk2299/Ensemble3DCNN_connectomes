import os
import numpy as np
import nibabel as nib
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import scale
from keras.layers import AveragePooling3D 
from keras.models import Sequential, Model
import keras

def cov_to_corr(covariance):
    diagonal = np.atleast_2d(1. / np.sqrt(np.diag(covariance)))
    correlation = covariance * diagonal * diagonal.T

    # Force exact 1. on diagonal
    np.fill_diagonal(correlation, 1.)
    return correlation

def get_correlation(data):
    
    cov_estimator=LedoitWolf(store_precision=False)
    return cov_to_corr(cov_estimator.fit(scale(data,with_mean=False, with_std=True)).covariance_)


def vcorrcoef(X,y):
    Xm = np.reshape(np.mean(X,axis=1),(X.shape[0],1))
    ym = np.mean(y)
    r_num = np.sum((X-Xm)*(y-ym),axis=1)
    r_den = np.sqrt(np.sum((X-Xm)**2,axis=1)*np.sum((y-ym)**2))
    r = r_num/r_den
    return r

def corr2_coeff(A,B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:,None]
    B_mB = B - B.mean(1)[:,None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1);
    ssB = (B_mB**2).sum(1);

    # Finally get corr coeff
    return np.dot(A_mA,B_mB.T)/np.sqrt(np.dot(ssA[:,None],ssB[None]))


def Downsample(input_size):
    model = Sequential()
    model.add(AveragePooling3D(pool_size=(2, 2, 2),input_shape = (
        61,73,61, input_size), border_mode='same'))
    return model

def get_fingerprint(subject_list, parcel_file, mask_file, data_folder, supervoxel = True):
    
    
    unit_data = nib.load(parcel_file).get_data()
    nodes = np.unique(unit_data).tolist()
    mask_data = nib.load(mask_file).get_data()
    
    if supervoxel:
        voxel_all=np.zeros(shape=[len(subject_list), 31,37,31,len(nodes)-1], dtype='float32') # AvgPool with downsampling factor 2 
        downsample = Downsample(voxel_all.shape[4])
    else:
        voxel_all=np.zeros(shape=[len(subject_list),61, 73, 61,len(nodes)-1], dtype='float32') # 3mm voxel resolution
    
    
    for i in range(len(subject_list)):
        print('Extracting connectivity fingerprint for subject:', subject_list[i])
        
        subject_folder = os.path.join(data_folder, subject_list[i])
        print('Subject folder', subject_folder)
        ro_file = [f for f in os.listdir(subject_folder) if f.endswith('_func_preproc.nii.gz')]
        data_file = os.path.join(subject_folder, ro_file[0])
        datafile = nib.load(data_file)
 
        img_data = datafile.get_data()
        sorted_list=[]
           
        for n in nodes:
                if n > 0:
                    node_array = img_data[unit_data == n]
                    avg = np.mean(node_array, axis=0)
                    avg = np.round(avg, 6)
                    sorted_list.append(avg.tolist())
                    
        
        sorted_list=np.asarray(sorted_list)
        
        img_data = datafile.get_data()[mask_data>0]
        img_data = img_data.reshape(-1,img_data.shape[1])
        
        labels_all=np.zeros(shape=[61*73*61,voxel_all.shape[4]])
        labels_all[(mask_data>0).reshape(-1)]=np.float32(corr2_coeff(img_data,sorted_list))
        labels_all[np.isnan(labels_all)]=0
        labels_all=np.reshape(labels_all, (1,61, 73, 61,voxel_all.shape[4]))
        
        if supervoxel:
            voxel_all[i] = downsample.predict(labels_all)[0]
        else:
            voxel_all[i] = labels_all[0]
        
    
    
    return voxel_all