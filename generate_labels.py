import os
import csv
import numpy as np

# Input variables
root_folder = '/home/mk2299/abide/abide_1035/ABIDE_pcp/'
phenotype_dict = {"abide1" : os.path.join(root_folder, 'Phenotypic_V1_0b_preprocessed1.csv'), "abide2": os.path.join(root_folder, 'ABIDEII_Composite_Phenotypic_1.csv')}

def get_subject_variable(subject_IDs, variable, version = "abide1"):
    var_dict = {}
    phenotype = phenotype_dict[version]
    with open(phenotype) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row['SUB_ID'] in subject_IDs:
                var_dict[row['SUB_ID']] = row[variable]

    return var_dict



def is_outlier(points, thresh=3.5):
  

    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score <= thresh

def good_frame_count(subject_list, motion_folder):
    
    remaining_frames = np.zeros(len(subject_list))
    
    for i in range(len(subject_list)):
        subject_file = os.path.join(motion_folder, subject_list[i], 'FD.1D')
        fd = np.genfromtxt(subject_file)
        bad_frames1 = np.union1d(np.where(fd>0.5)[0]-1,np.where(fd>0.5)[0]+1)  # Removing 1 frame before and 1 frame after
        bad_frames2 = np.union1d(np.where(fd>0.5)[0],np.where(fd>0.5)[0]+2)   #Removing current frame and 2 frames after
        bad_frames=np.union1d(bad_frames1,bad_frames2)
        bad_frames=[x for x in bad_frames if x not in [-1,len(fd)]]   #Boundary condition
        remaining_frames[i]=len(fd)-len(bad_frames)
    return remaining_frames
 
def get_dx_labels(subject_IDs, version = "abide1"):
    dx = get_subject_variable(subject_IDs, 'DX_GROUP', version)
    
    
    dx_labels = np.zeros(len(subject_IDs))
    
    for i in range(len(subject_IDs)):
        dx_labels[i]=dx[subject_IDs[i]]
        
        
    for i in range(len(dx_labels)):
        if dx_labels[i]==1:
            dx_labels[i]=0
        if dx_labels[i]==2:
            dx_labels[i]=1

    
    return dx_labels         

def get_age_labels(subject_IDs, version = "abide1"):
    if version=="abide1":
        age = get_subject_variable(subject_IDs, 'AGE_AT_SCAN', version)
    else:
        age = get_subject_variable(subject_IDs, 'AGE_AT_SCAN ', version) 
    
    age_labels = np.zeros(len(subject_IDs))
    
    for i in range(len(subject_IDs)):
        age_labels[i] = age[subject_IDs[i]]
         
    return age_labels 
                  
def get_reduced_data(subject_IDs): 
    
    dx = get_subject_variable(subject_IDs, 'DX_GROUP')
    age_at_scan = get_subject_variable(subject_IDs, 'AGE_AT_SCAN')
    site_id = get_subject_variable(subject_IDs, 'SITE_ID')
    
    dx_labels = np.zeros(len(subject_IDs))
    age_all = np.zeros(len(subject_IDs))
    
    sites_all=[]
    
    for i in range(len(subject_IDs)):
        
        
        dx_labels[i] = dx[subject_IDs[i]]
        age_all[i] = age_at_scan[subject_IDs[i]]
        sites_all.append(site_id[subject_IDs[i]])
        
    for i in range(len(dx_labels)):
        if dx_labels[i]==1:
            dx_labels[i]=0
        if dx_labels[i]==2:
            dx_labels[i]=1

    
    return dx_labels, age_all, np.array(sites_all)