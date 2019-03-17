import os 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from keras import optimizers
from keras import backend as K
from keras import backend as be
from keras import optimizers
from keras.utils.training_utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint
from keras import losses
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.cross_validation import StratifiedKFold
from keras import regularizers
import numpy as np
import keras
import _pickle as cPickle
from preprocessing import get_fingerprint
from generate_labels import get_dx_labels, get_reduced_data
import models_3DCNN
import argparse

def train_and_evaluate_model(train, test, x, y, params, average = True): 

    filepath = "temp_weights/weights.{epoch:02d}.hdf5"
    
    save_all_mod = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only = False, save_weights_only = True)
 
    model = models_3DCNN.Classifier3DCNN(x.shape[1:])    ## Load in the regression model 
    if params['gpu_count'] > 1:
        model = multi_gpu_model(model, gpus= args.gpu_count)
        
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.SGD(momentum=0.9, lr = params['lrate'], decay=1e-3), metrics=['accuracy'])
    
    #Train a single fold of the data
    model.fit(x[train],y[train], callbacks=[save_all_mod], shuffle = True, 
          batch_size = params['batch_size'], epochs = params['epochs'])
     
    
    predictions = model.predict(x[test], verbose=0)[:,0]
    scores = mse(predictions, y[test])
    print('Averaged model score: ', scores)
    K.clear_session()
    
    for i in range(100):
        os.remove("temp_weights/weights." + str("{:02d}".format(i))+".hdf5")
  
    return scores

def model_results_abide1(X_data, Y_data, params):
    
    accuracy_all=[]
    
    print('Running model..')
    
    for i, (train, test) in enumerate(params['skf']):
     
        print("Running Fold", i+1, "/", params['n_folds'])
        res = train_and_evaluate_model(train, test, X_data, Y_data, params)
        accuracy_all.append(res)
  
    
    K.clear_session()
    be.clear_session()
    print('Mean 10-fold accuracy: ', np.mean(accuracy_all))
    return 


def model_results_abide2(X_train, Y_train, X_test, Y_test, params):
    
    filepath = "temp_weights/weights.{epoch:02d}.hdf5"
    save_all_mod = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only = False, save_weights_only=True) 
    
    model = models_3DCNN.Classifier3DCNN(X_train.shape[1:])   ## Load in the regression model 
   
    if params['gpu_count'] > 1:
        model = multi_gpu_model(model, gpus= args.gpu_count)
        
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.SGD(momentum=0.9, lr = params['lrate'], decay=1e-3), metrics=['accuracy'])
     
    model.fit(X_train, Y_train, callbacks=[save_all_mod], 
          batch_size = params['batch_size'], epochs = params['batch_size'])
       
    
    Y_pred = model.predict(X_test, verbose=0)[:,0]
    
    print('Averaged model error: ', mse(Y_pred, Y_test))
   
    model.save_weights('model.h5')
    
    for i in range(100):
        os.remove("temp_weights/weights." + str("{:02d}".format(i))+".hdf5")
    
    return 

def main():
    parser = argparse.ArgumentParser(description='3D-CNN for classification of functional connectomes')
    parser.add_argument('--abide_version', default='2', type=float, help='Run training on ABIDE-I to predict results on ABIDE-II or run 10-fold cross-validation on ABIDE-I')
    parser.add_argument('--lrate', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default = 64, type=int)
    parser.add_argument('--parcel_scale', default = 110, type=int, help ='Number of ROIs..should be 110, 160, 200, or 400')
    parser.add_argument('--parcel_index', default = 1, type=int, help ='Index of parcellation (Must be between 1 and 30 since there are 30 parcellations at each scale)')
    parser.add_argument('--preprocessed_abide1_file', default = '/path/to/data', help='File name of .npy file containing connectivity fingerprints of ABIDE-I data')
    parser.add_argument('--preprocessed_abide2_file', default = '/path/to/data', help='File name of .npy file containing connectivity fingerprints of ABIDE-II data')
    parser.add_argument('--train_list_file', default= 'subjects_ABIDEI.txt', help = '.txt file containing subject IDs of training subjects')
    parser.add_argument('--test_list_file', default = 'subjects_ABIDEII.txt', help = '.txt file containing subject IDs of test subjects')
    parser.add_argument('--gpu_count', default =1, help = 'Number of GPUs to use during training')
    parser.add_argument('--mask_file', default = './misc/grey_matter_mask.nii.gz', help ='Gray matter mask to use for extracting connectivity fingerprints')
    parser.add_argument('--data_folder', default = None, help = 'root folder for data')
    parser.add_argument('--folds', default = None, help = 'Pickle file containing folds of ABIDE-I')
    args = parser.parse_args()
    params = dict()
    params['lrate'] = args.lrate                    
    params['epochs'] = args.epochs  
    params['batch_size'] = args.batch_size
    params['parcel_scale'] = args.parcel_scale    
    params['parcel_index'] = args.parcel_index
    params['gpu_count'] = args.gpu_count
    params['n_folds'] = 10 #Default
    
    parcellation_file = os.path.join('./parcellation_masks', str(args.parcel_scale),'SP_' + str(args.parcel_index) +'.nii.gz') 
  
    train_list = np.genfromtxt(args.train_list_file, dtype=str)
    test_list = np.genfromtxt(args.test_list_file, dtype=str)
    
    Y_train = get_dx_labels(train_list)
    Y_test = get_dx_labels(test_list, version="abide2")
    
    
    ## Extracting training data from ABIDE-I
    if os.path.isfile(args.preprocessed_abide1_file):
        X_train = np.load(args.preprocessed_abide1_file)
    else:
        X_train = get_fingerprint(train_list, parcellation_file, args.mask_file, args.data_folder)
    
    
    ## Run either 10-fold cross-validation on ABIDE-I or train on ABIDE-I entirely/test on ABIDE-II
    if args.abide_version == 1:
        if args.folds:
            skf = cPickle.load(open(args.folds, "rb" ))
            params['skf'] = skf
        else:
            site_names, age, labels =  get_reduced_data(train_list)
            classes = ['{0}{1}'.format(site_name, asd_label) for site_name, asd_label in zip(site_names, labels)] 
            
            ## Create stratified folds using diagnosis labels and sites
            skf = StratifiedKFold(classes, n_folds=10, shuffle=True) 
            params['skf'] = skf
        model_results_abide1(X_train, Y_train, params)
    else:
        
        if os.path.isfile(args.preprocessed_abide2_file):
            X_test = np.load(args.preprocessed_abide1_file)
        else:
            X_test = get_fingerprint(test_list, parcellation_file, args.mask_file, args.data_folder)
        model_results_abide2(X_train, Y_train, X_test, Y_test, params)   
 

if __name__ == '__main__':
    main()