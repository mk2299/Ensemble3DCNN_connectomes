# Ensemble learning with 3D convolutional neural networks for functional connectome-based prediction

This code provides a python Keras based implementation of the following paper: \
Meenakshi Khosla, Keith Jamison, Amy Kuceyeski,c, Mert R. Sabuncu \
"Ensemble learning with 3D convolutional neural networks for connectome-based prediction" \
Arxiv: https://arxiv.org/abs/1809.06219

keywords: Functional connectivity, fMRI, Convolutional Neural Networks,
Autism Spectrum Disorder, ABIDE

This implementation is catered to the ABIDE dataset where we focus on two prediction tasks: \
(A) A regression model for age prediction (See train_and_evaluate_regression.py) \
(B) A classification model for ASD/HC Classification (See train_and_evaluate_classification.py) 


# Instructions 


__Setup__  \
Set up the path for the Phenotyptic files in generate_labels.py in accordance with your directory structure. 
This method works with volumetric nifti files (.nii.gz). You will have to change the file naming in preprocessing.py if you are working with preprocessed functional files not ending in '_func_preproc.nii.gz'.   

__Requirements__ \
Tensorflow (>1.3.0) \
Keras (=2.0.8) \
NumPy \
NiBabel \
Scikit-learn \

__Usage__ \
To learn about the parameters, use \
python train_and_evaluate_regression.py --help \
OR python train_and_evaluate_classification.py --help 

To run the models with default parameters, run \
python train_and_evaluation_regression.py --data_folder /path/to/data \
OR python train_and_evaluation_classification.py --data_folder /path/to/data



