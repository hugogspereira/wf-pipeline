# Code for the paper Tik-Tok: The Utility of Packet Timing in Website Fingerprinting Attacks accepted in PETS 2020.
# Mohammad Saidur Rahman - saidur.rahman@mail.rit.edu
# Global Cybersecurity Institute, Rochester Institute of Technology

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from final_features_process import *
from DF_Model import *
import os
import numpy as np

# Use modern TensorFlow/Keras imports
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from os import path
import argparse
import random
random.seed(583004949)

# Suppress TensorFlow warnings
import warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

################################################################################
# TODO: Adjust these based on your dataset
dataset = 'Undefended'
num_classes = 1000 # Set number of classes for your dataset

print(f"Processing dataset: {dataset}")
print(f"Number of classes: {num_classes}")

data_root = '../../../data/features/'
save_path = os.getcwd() + '/' + 'save_data/' + str(dataset) + '/'

try:
    os.stat(save_path)
except:
    os.makedirs(save_path)

print(f"Data root: {data_root}")
print(f"Save path: {save_path}")

# Check whether the files for training the model already exists
model_files = ['X_tr', 'Y_tr', 'X_vl', 'Y_vl', 'X_te', 'Y_te']
count_m_files = 0
for f in model_files:
    f_path = save_path + f + '.pkl'
    if path.exists(f_path):
        count_m_files += 1

print(f"Found {count_m_files}/6 processed model files")

if count_m_files == 6:
    print("Loading pre-processed data...")
    X_train, y_train, X_valid, y_valid, X_test, y_test = final_data_load(save_path)
else:
    print("Processing raw data (this may take a while)...")
    try:
        X_train, y_train, X_valid, y_valid, X_test, y_test = final_process(dataset, data_root, save_path)
    except Exception as e:
        print(f"Error processing data: {e}")
        print("Make sure your data is in the correct directory structure at:", data_root)
        exit(1)

# Convert data as float32 type
X_train = X_train.astype('float32')
X_valid = X_valid.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('float32')
y_valid = y_valid.astype('float32')
y_test = y_test.astype('float32')

# we need a [Length x 1] x n shape as input to the DF CNN (Tensorflow)
X_train = X_train[:, :, np.newaxis]
X_valid = X_valid[:, :, np.newaxis]
X_test = X_test[:, :, np.newaxis]

print(X_train.shape[0], 'train samples')
print(X_valid.shape[0], 'validation samples')  
print(X_test.shape[0], 'test samples')

# Verify data distribution
print(f"Unique classes in y_train: {len(np.unique(y_train))}")
print(f"Expected num_classes: {num_classes}")
if len(y_train) > 0:
    print(f"Samples per class (approx): {len(y_train) // num_classes}")

# Convert class vectors to categorical classes matrices
y_train = to_categorical(y_train, num_classes)
y_valid = to_categorical(y_valid, num_classes)
y_test = to_categorical(y_test, num_classes)

print(f"Input shape: {X_train.shape}")
print(f"Output shape: {y_train.shape}")

run_trial = 1  # change run_trial > 1 to get a standard deviation of the accuracy.
seq_length = 160  # 8 timing features x 20 bins = 160 features values.
num_epochs = 100  # 100 epochs for experiments with timing_features and onion_sites
VERBOSE = 2

print(f"Running {run_trial} trial(s) with {num_epochs} epochs each")
print(f"Sequence length: {seq_length}")

df_res = [None] * run_trial
for j in range(run_trial):
    print(f"Starting trial {j+1}/{run_trial}")
    df_res[j] = df_accuracy(num_classes, num_epochs, seq_length, VERBOSE, X_train, y_train, X_valid, y_valid, X_test, y_test)
    print(f"Trial {j+1} accuracy: {df_res[j]:.4f}")

if run_trial != 1:
    print(f'Mean Acc: {np.mean(df_res):.4f}')
    print(f'STD of Mean: {np.std(df_res):.4f}')
else:
    print(f'Final Accuracy: {df_res[0]:.4f}')