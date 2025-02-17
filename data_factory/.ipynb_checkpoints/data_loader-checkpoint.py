import torch
import os
import random
from torch.utils.data import Dataset, Subset
from torch.utils.data import DataLoader
import numpy as np
import collections
import numbers
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
import pickle


class SWaTSegLoader(Dataset):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(data_path + '/train.csv', header=1)
        data = data.values[:, 1:-1]

        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)

        test_data = pd.read_csv(data_path + '/test.csv')

        y = test_data['Normal/Attack'].to_numpy()
        labels = []
        for i in y:
            if i == 'Attack':
                labels.append(1)
            else:
                labels.append(0)
        labels = np.array(labels)


        test_data = test_data.values[:, 1:-1]
        test_data = np.nan_to_num(test_data)

        self.test = self.scaler.transform(test_data)
        self.train = data
        self.test_labels = labels.reshape(-1, 1)

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        mode : "train" or "test"
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.train.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])


class PSMSegLoader(Dataset):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(data_path + '/train.csv')
        data = data.values[:, 1:]

        data = np.nan_to_num(data)

        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(data_path + '/test.csv')

        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)

        self.test = self.scaler.transform(test_data)

        self.train = data

        self.test_labels = pd.read_csv(data_path + '/test_label.csv').values[:, 1:]

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        mode : "train" or "test"
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.train.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        


class _WACASegLoader(Dataset):
    """ Updated for scaling, includes timestamps """
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        
        # Read raw data (including timestamp)
        data = pd.read_csv(data_path + '/train.csv')
        
        self.timestamps_train = data.iloc[:, 1].values
        data = data.values[:, [0, 2, 3, 4]]  # Keep sensor, x, y, z columns only
        data = np.nan_to_num(data)

        # Compute min/max for each sensor and feature (x, y, z) from train data 
        self.sensor_min_max = self.compute_sensor_min_max(data)

        # Now scale the training data (excluding timestamp)
        scaled_data = self.min_max_scale(data, self.sensor_min_max)
        
        # Scale timestamps using StandardScaler
        self.timestamp_scaler = StandardScaler().fit(self.timestamps_train.reshape(-1, 1))
        scaled_timestamps_train = self.timestamp_scaler.transform(self.timestamps_train.reshape(-1, 1))

        
         # Reattach timestamps to scaled data (place timestamp back as the first column)
        self.train = np.column_stack((scaled_timestamps_train, scaled_data[:, 1:]))
        print("Train data with ts", self.train[:5])
        print("Training scale complete")

        # Read and scale the test data using the same min/max from training
        test_data = pd.read_csv(data_path + '/test.csv')
        
        self.timestamps_test = test_data.iloc[:, 1].values
        test_data = test_data.values[:, [0, 2, 3, 4]]  # Keep sensor, x, y, z columns only
        test_data = np.nan_to_num(test_data)
        
        # scale the test data (excluding timestamp)
        self.test = self.min_max_scale(test_data, self.sensor_min_max)
        
        # Scale timestamps using the same StandardScaler
        scaled_timestamps_test = self.timestamp_scaler.transform(self.timestamps_test.reshape(-1, 1))
        
        self.test = np.column_stack((scaled_timestamps_test, self.test[:, 1:]))
        print("test data:", self.test[:5])

        # Read test labels
        self.test_labels = pd.read_csv(data_path + '/test_label.csv').values[:, 1:]
        
        print("test:", self.test.shape)
        print("train:", self.train.shape)


    def compute_sensor_min_max(self, data):
        """
        Compute the min and max for each feature (x, y, z) for each sensor from the raw data.
        
        Returns:
            dict: Min/max values for each feature for each sensor.
        """
        # Initialize the dictionary for storing min/max values
        sensor_min_max = {}
        # Extract columns: sensor labels, x, y, z (assuming sensor label is in column 1)
        sensor_labels = data[:, 0].astype(int)  # Assuming sensor label is in column 1
        x_values = data[:, 1]
        y_values = data[:, 2]
        z_values = data[:, 3]

        # Compute min/max for each sensor
        for sensor_label in np.unique(sensor_labels):
            sensor_label = int(sensor_label)
            sensor_mask = sensor_labels == sensor_label
            sensor_x = x_values[sensor_mask]
            sensor_y = y_values[sensor_mask]
            sensor_z = z_values[sensor_mask]

            # Store min and max for each feature (x, y, z) for the current sensor
            sensor_min_max[sensor_label] = {
                'x': (sensor_x.min(), sensor_x.max()),
                'y': (sensor_y.min(), sensor_y.max()),
                'z': (sensor_z.min(), sensor_z.max())
            }

        return sensor_min_max

    def min_max_scale(self, data, sensor_min_max):
        """
        Apply min-max scaling to the data using the provided sensor min/max values.
        
        Args:
            data (numpy array): The raw data to scale.
            sensor_min_max (dict): Dictionary containing min/max values for each feature for each sensor.
        
        Returns:
            numpy array: The scaled data.
        """
        scaled_data = data.copy()
        
        # Extract columns: sensor labels, x, y, z (assuming sensor label is in column 0)
        sensor_labels = data[:, 0].astype(int)  # Assuming sensor label is in column 0
        x_values = data[:, 1]
        y_values = data[:, 2]
        z_values = data[:, 3]

        # Apply scaling for each sensor (excluding timestamp)
        for sensor_label in np.unique(sensor_labels):
            print("Unique sensor labels:", np.unique(sensor_labels))
            sensor_label = int(sensor_label)
            # Get min and max for each feature of this sensor
            min_x, max_x = sensor_min_max[sensor_label]['x']
            min_y, max_y = sensor_min_max[sensor_label]['y']
            min_z, max_z = sensor_min_max[sensor_label]['z']

            # Scale the data for the current sensor
            sensor_mask = sensor_labels == sensor_label
            scaled_data[sensor_mask, 1] = (x_values[sensor_mask] - min_x) / (max_x - min_x)
            scaled_data[sensor_mask, 2] = (y_values[sensor_mask] - min_y) / (max_y - min_y)
            scaled_data[sensor_mask, 3] = (z_values[sensor_mask] - min_z) / (max_z - min_z)

        return scaled_data
    
    def __len__(self):
        """
        Number of images in the object dataset.
        mode : "train" or "test"
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.train.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        
class WACASegLoader(Dataset):
    """
    WACASegLoader implementation from AnomalyTransformer. Does not include timestamps. 
    Uses RobustScaler for the training data, and min/max of the training data to scale the test data.
    """
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        
        # Read raw data (lightly cleaned before added to train.csv, data sorted and insignificant sensors removed)
        raw_data = pd.read_csv(data_path + '/train.csv', header=0)
        
        # First discard 2.5 - 5 seconds from start and end of the timeseries 
        # Corresponds to user putting on/taking off the smartwatch
        data = raw_data.iloc[250:-250].astype("float32").reset_index(drop=True)  # Reset index to keep it clean
        
        # Compute min/max for each sensor and feature (x, y, z) from train data 
        self.sensor_min_max = self.compute_sensor_min_max(data.values[:, [0, 2, 3, 4]]) # disregards timestamps

        # Scale the training data using sklearn RobustScaler 
        ## Scaling should be done on the xyz axes, but not the sensors 
        
        # Step 1:split data into separate dfs for each sensor
        sensor_dfs = {}
        for sensor in data['sensor'].unique():
            sensor_dfs[sensor] = data[data['sensor'] == sensor].copy()

        # Step 2: Scale on each sensor
        for sensor, df_sensor in sensor_dfs.items():
            scaler = RobustScaler()
            # Fit and transform only the x, y, z columns.
            scaled_features = scaler.fit_transform(df_sensor[['x', 'y', 'z']])
            df_sensor[['x', 'y', 'z']] = scaled_features
            sensor_dfs[sensor] = df_sensor

        # Step 3: Concatenate the scaled DataFrames and sort them back into chronological order
        scaled_data = pd.concat(sensor_dfs.values()).sort_values(by="timestamp").reset_index(drop=True)
        scaled_data = scaled_data.drop(columns=["timestamp"])
        
        
        # Split into train and validation sets
        data_len = len(scaled_data)
        self.train = scaled_data[:int(data_len * 0.8)]
        self.train = self.train.to_numpy()
        self.val = scaled_data[int(data_len * 0.8):]
        self.val = self.val.to_numpy()
        
        print("train", self.train[:5])
        
        # Read and scale the test data
        test_data = pd.read_csv(data_path + '/test.csv')
        test_data = test_data.values[:, [0, 2, 3, 4]].astype("float32")  # Keep sensor, x, y, z columns only
        test_data = np.nan_to_num(test_data)
        self.test = self.min_max_scale(test_data, self.sensor_min_max)
        print("test", self.test[:5])

        # Read test labels
        self.test_labels = pd.read_csv(data_path + '/test_label.csv').values[:, 1:]
        
        print("test:", self.test.shape)
        print("train:", self.train.shape)
        print("val:", self.val.shape)

    def compute_sensor_min_max(self, data):
        """
        Compute the min and max for each feature (x, y, z) for each sensor from the raw data.
        
        Returns:
            dict: Min/max values for each feature for each sensor.
        """
        # Initialize the dictionary for storing min/max values
        sensor_min_max = {}
        # Extract columns: sensor labels, x, y, z (assuming sensor label is in column 1)
        sensor_labels = data[:, 0].astype(int)  # Assuming sensor label is in column 1
        x_values = data[:, 1]
        y_values = data[:, 2]
        z_values = data[:, 3]

        # Compute min/max for each sensor
        for sensor_label in np.unique(sensor_labels):
            sensor_label = int(sensor_label)
            sensor_mask = sensor_labels == sensor_label
            sensor_x = x_values[sensor_mask]
            sensor_y = y_values[sensor_mask]
            sensor_z = z_values[sensor_mask]

            # Store min and max for each feature (x, y, z) for the current sensor
            sensor_min_max[sensor_label] = {
                'x': (sensor_x.min(), sensor_x.max()),
                'y': (sensor_y.min(), sensor_y.max()),
                'z': (sensor_z.min(), sensor_z.max())
            }

        return sensor_min_max

    def min_max_scale(self, data, sensor_min_max):
        """
        Apply min-max scaling to the data using the provided sensor min/max values.
        
        Args:
            data (numpy array): The raw data to scale.
            sensor_min_max (dict): Dictionary containing min/max values for each feature for each sensor.
        
        Returns:
            numpy array: The scaled data.
        """
        scaled_data = data.copy()
        
        # Extract columns: sensor labels, x, y, z (assuming sensor label is in column 0)
        sensor_labels = data[:, 0].astype(int)  # Assuming sensor label is in column 0
        x_values = data[:, 1]
        y_values = data[:, 2]
        z_values = data[:, 3]

        # Apply scaling for each sensor (excluding timestamp)
        for sensor_label in np.unique(sensor_labels):
            print("Unique sensor labels:", np.unique(sensor_labels))
            sensor_label = int(sensor_label)
            # Get min and max for each feature of this sensor
            min_x, max_x = sensor_min_max[sensor_label]['x']
            min_y, max_y = sensor_min_max[sensor_label]['y']
            min_z, max_z = sensor_min_max[sensor_label]['z']

            # Scale the data for the current sensor
            sensor_mask = sensor_labels == sensor_label
            scaled_data[sensor_mask, 1] = (x_values[sensor_mask] - min_x) / (max_x - min_x)
            scaled_data[sensor_mask, 2] = (y_values[sensor_mask] - min_y) / (max_y - min_y)
            scaled_data[sensor_mask, 3] = (z_values[sensor_mask] - min_z) / (max_z - min_z)

        return scaled_data
    
    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'val':
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index: index+self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.mode == 'val':
            return np.float32(self.val[index: index+self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.mode == 'test':
            return np.float32(self.test[index: index+self.win_size]), np.float32(
                self.test_labels[index: index+self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])

        
class MSLSegLoader(Dataset):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/MSL_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/MSL_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.test_labels = np.load(data_path + "/MSL_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.train.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])

class SMAPSegLoader(Dataset):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMAP_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMAP_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.test_labels = np.load(data_path + "/SMAP_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.train.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])

class SMDSegLoader(Dataset):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMD_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMD_test.npy")
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.test_labels = np.load(data_path + "/SMD_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)
        
    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.train.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])


def get_loader_segment(data_path, batch_size, win_size=100, step=100, mode='train', dataset='KDD', val_ratio=0.2):
    '''
    model : 'train' or 'test'
    '''
    if (dataset == 'SMD'):
        dataset = SMDSegLoader(data_path, win_size, step, mode)
    elif (dataset == 'MSL'):
        dataset = MSLSegLoader(data_path, win_size, step, mode)
    elif (dataset == 'SMAP'):
        dataset = SMAPSegLoader(data_path, win_size, step, mode)
    elif (dataset == 'PSM'):
        dataset = PSMSegLoader(data_path, win_size, step, mode)
    elif (dataset == 'SWaT'):
        dataset = SWaTSegLoader(data_path, win_size, step, mode)
    elif (dataset == 'WACA'):
        dataset = WACASegLoader(data_path, win_size, step, mode)

    shuffle = False
    if mode == 'train':
        shuffle = True

        dataset_len = int(len(dataset))
        train_use_len = int(dataset_len * (1 - val_ratio))

        val_use_len = int(dataset_len * val_ratio)
        val_start_index = random.randrange(train_use_len)


        indices = torch.arange(dataset_len)
        

        train_sub_indices = torch.cat([indices[:val_start_index], indices[val_start_index+val_use_len:]])
        train_subset = Subset(dataset, train_sub_indices)

        val_sub_indices = indices[val_start_index:val_start_index+val_use_len]
        val_subset = Subset(dataset, val_sub_indices)
        
        train_loader = DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
        val_loader = DataLoader(dataset=val_subset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

        k_use_len = int(train_use_len*0.1)
        k_sub_indices = indices[:k_use_len]
        k_subset = Subset(dataset, k_sub_indices)
        k_loader = DataLoader(dataset=k_subset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

        return train_loader, val_loader, k_loader

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)
    
    return data_loader, data_loader