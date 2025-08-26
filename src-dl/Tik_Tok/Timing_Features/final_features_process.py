import pandas as pd
import numpy as np
import pickle
from sklearn.utils import shuffle
from common import *
from features import *
import os
import time
import glob
import random
random.seed(583004949)

################################################################################
# Constants - TODO: Adjust these based on your dataset
num_sites = 1000
bin_size = 20

################################################################################
# Function to get all available samples for a given site by checking existing files
def get_available_samples(data_path, site):
    """
    Get all available sample numbers for a given site by checking existing files
    """
    available_samples = []
    pattern = f"{site}-*"
    
    # Use glob to find all matching files
    file_pattern = os.path.join(data_path, pattern)
    matching_files = glob.glob(file_pattern)
    
    for file_path in matching_files:
        filename = os.path.basename(file_path)
        # Extract sample number
        parts = filename.split('-')
        if len(parts) >= 2:
            try:
                sample_num = int(parts[1])
                available_samples.append(sample_num)
            except ValueError:
                continue
    
    return sorted(available_samples)

################################################################################
# Function to generate and save features for training, validation, and testing datasets
def gen_save_feats(dataset, data_path, save_path):

    for i in range(3):
        print('Iteration: ', i)
        features = {
            "MED": {},
            "IBD_FF": {},
            "IBD_IFF": {},
            "IBD_LF": {},
            "IBD_OFF": {},
            "Burst_Length": {},
            "IMD": {},
            "Variance": {},
        }
        
        if i == 0:
            print('Processing Training Data ...')
        elif i == 1:
            print('Processing Validation Data ...')
        else:
            print('Processing Testing Data ...')
        
        labels_instances = []
        
        for site in range(1, num_sites+1):
            # Get all available samples for this site
            available_samples = get_available_samples(data_path, site)
            
            if len(available_samples) == 0:
                # Debugging line to check if no samples are found
                # print(f"Warning: No samples found for site {site}")
                continue
            
            # Determine which samples to use for this iteration
            total_samples = len(available_samples)
            
            if i == 0:  # Training
                end_idx = int(0.8 * total_samples)  # Use first 80% of available samples
                selected_samples = available_samples[:end_idx]
            elif i == 1:  # Validation
                start_idx = int(0.8 * total_samples)
                end_idx = int(0.9 * total_samples)  # Use next 10% of available samples
                selected_samples = available_samples[start_idx:end_idx]
            else:  # Testing
                start_idx = int(0.9 * total_samples) # Use last 10% of available samples
                selected_samples = available_samples[start_idx:]
            
            # Process selected samples
            for sample_num in selected_samples:
                final_fname = str(site) + "-" + str(sample_num)
                file_path = os.path.join(data_path, final_fname)
                
                # Check if file exists before trying to open it
                if not os.path.exists(file_path):
                    print(f"Warning: File {file_path} not found, skipping...")
                    continue
                
                try:
                    # Directory of the raw data
                    with open(file_path, "r") as file_pt:
                        traces = []
                        for line in file_pt:
                            x = line.strip().split('\t')
                            x[0] = float(x[0])
                            x[1] = 1 if float(x[1]) > 0 else -1
                            traces.append(x)
                        
                        bursts, direction_counts = extract_bursts(traces)
                        features["MED"][final_fname] = MED(bursts)
                        features["IBD_FF"][final_fname] = IBD_FF(bursts)
                        features["IBD_IFF"][final_fname] = IBD_IFF(bursts)
                        features["IBD_LF"][final_fname] = IBD_LF(bursts)
                        features["IBD_OFF"][final_fname] = IBD_OFF(bursts)
                        features["Burst_Length"][final_fname] = Burst_Length(bursts)
                        features["IMD"][final_fname] = IMD(bursts)
                        features["Variance"][final_fname] = Variance(bursts)
                        labels_instances.append(final_fname)
                        
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
                    continue
            
        feature_bins = {
            "MED": bin_size,
            "IBD_FF": bin_size,
            "IBD_IFF": bin_size,
            "IBD_LF": bin_size,
            "IBD_OFF": bin_size,
            "Burst_Length": bin_size,
            "IMD": bin_size,
            "Variance": bin_size,
        }
        
        # Create bins for each feature, extract bin counts and normalize them
        if i == 0:
            print ("Extracting Training Features ...")
            output_file = 'training'
        elif i == 1:
            print ("Extracting Validation Features ...")
            output_file = 'validation'
        else:
            print ("Extracting Testing Features ...")
            output_file = 'testing'

        for feature in features:
            features[feature] = normalize_data(features[feature], feature_bins[feature])

        feature_names = features.keys()

        if i == 0:
            print ("Saving Training Features ...")
        elif i == 1:
            print ("Saving Validation Features ...")
        else:
            print ("Saving Testing Features ...")
            
        with open(save_path + output_file, "w") as out:
            for label in labels_instances:
                data = []
                data.extend(values
                            for f in feature_names
                            for values in features[f][label])

                site_number = int(label.split("-")[0])
                class_label = site_number - 1 # zero-indexed labels
                row = ",".join([str(val) for val in data]) + "," + str(class_label) + "\n"
                out.write(row)
        print(f'Done with iteration {i}, processed {len(labels_instances)} samples')

################################################################################
# Function to read data from a CSV file
def read_data(file_name):
    data = pd.read_csv(file_name, header=None)
    return data[data.columns[:-1]].values, \
        data[data.columns[-1]].values

################################################################################
# Function to create data matrix and save to pickle files
def making_matrx(X, Y, f_name, f_dir):
    m, n = X.shape
    
    X_ = np.zeros(shape=(m, n))
    Y_ = np.zeros(shape=(m,))

    labels = np.unique(Y)
    ind1 = 0

    for i in np.arange(labels.size):
        indices = np.where(Y == labels[i])[0]

        splt_nbr = int(round(indices.size))
        X_[ind1:ind1+splt_nbr, :] = X[indices[:splt_nbr], :]
        Y_[ind1:ind1+splt_nbr] = Y[indices[:splt_nbr]]

        ind1 += splt_nbr

    X_final, Y_final = shuffle(X_, Y_)
    
    # Dumping X_train, Y_train, X_test, Y_test, X_val, Y_val in pickle files
    data_dir_out = f_dir
    pickle_file_X = data_dir_out + 'X_' + f_name + '.pkl'
    with open(pickle_file_X, 'wb') as handle:
        pickle.dump(X_final, handle, protocol=pickle.HIGHEST_PROTOCOL)

    pickle_file_Y = data_dir_out + 'Y_' + f_name + '.pkl'
    with open(pickle_file_Y, 'wb') as handle:
        pickle.dump(Y_final, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #print "Pickle Files Done!!..................."
    return X_final, Y_final

################################################################################
# Main function to process data and return training, validation, and testing sets
def final_process(dataset, data_root, save_path):
    st_time = time.time()
    print('Processing ', dataset,' data.')
    
    gen_save_feats(dataset, data_root, save_path)
    print('Features Processing Completed in ', (time.time() - st_time)/60, ' mins.')
    
    train_file = 'training'
    valid_file = 'validation'
    test_file = 'testing'
    
    X_tr, Y_tr = read_data(save_path + train_file)
    X_vl, Y_vl = read_data(save_path + valid_file)
    X_te, Y_te = read_data(save_path + test_file)
    
    X_train, y_train = making_matrx(X = X_tr, Y = Y_tr, f_name = 'tr', f_dir = save_path)
    X_valid, y_valid = making_matrx(X = X_vl, Y = Y_vl, f_name = 'vl', f_dir = save_path)
    X_test, y_test = making_matrx(X = X_te, Y = Y_te, f_name = 'te', f_dir = save_path)
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test
    
################################################################################
# Function to load processed data from pickle files
def final_data_load(save_path):
    with open(save_path + 'X_tr.pkl', 'rb') as handle:
        X_train = pickle.load(handle)
    with open(save_path + 'Y_tr.pkl', 'rb') as handle:
        y_train = pickle.load(handle)
    with open(save_path + 'X_vl.pkl', 'rb') as handle:
        X_valid = pickle.load(handle)
    with open(save_path + 'Y_vl.pkl', 'rb') as handle:
        y_valid = pickle.load(handle)
    with open(save_path + 'X_te.pkl', 'rb') as handle:
        X_test = pickle.load(handle)
    with open(save_path + 'Y_te.pkl', 'rb') as handle:
        y_test = pickle.load(handle)
        
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_valid = np.array(X_valid)
    y_valid = np.array(y_valid)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    return X_train, y_train, X_valid, y_valid, X_test, y_test