import pandas as pd
import numpy as np


def clean_data(filepath):
    column_names = ["sensor", "timestamp", "x", "y", "z"]
    column_types = {
        "sensor": "int16",      # Enforce integer type for sensor
        "timestamp": "float32", # Enforce float type for timestamp
        "x": "float32",         # Enforce float type for x
        "y": "float32",         # Enforce float type for y
        "z": "float32"          # Enforce float type for z
    }

    try:
        # Read the file and skip rows with bad lines
        df = pd.read_csv(filepath, 
                         on_bad_lines="skip", 
                         names=column_names, 
                         dtype=column_types)

        # Include only gyro (10) and accel (4) sensor types
        df = df[df["sensor"].isin([4, 10])]

        # Sort by timestamp
        df = df.sort_values(by="timestamp")

        return df

    except ValueError as e:
        # print(f"Error reading file {filepath}: {e}")
        return None

def has_sufficient_data(df):
    # Filter accelerometer and gyroscope data
    acc_count = df[df['sensor'] == 4].shape[0]  # Sensor type '4' for accelerometer
    gyro_count = df[df['sensor'] == 10].shape[0]  # Sensor type '10' for gyroscope

    # Check if both have at least 15 data points
    return acc_count >= 15 and gyro_count >= 15


def get_valid_users(num_users, typing_task):
    valid_users = []
    for user in range(1, 50):
        current_user = f"user{user}"
        path_to_train = f"data/WACA/WACA/WACA_dataset/{current_user}_{typing_task}.csv"
        if not os.path.exists(path_to_train): continue

        df = clean_data(path_to_train)
        if df is None:
            continue

        sufficient_count = has_sufficient_data(df)
        if sufficient_count:
            valid_users.append(current_user)
        else:
            print(f"Lacking sufficient sensor data: {current_user}")
    return valid_users

def save_user_train_data(path_to_train, path_to_npy="data/WACA/WACA/WACA_train.npy", path_to_csv="data/WACA/WACA/train.csv"):
    df = clean_data(path_to_train)
    if has_sufficient_data(df):
        np.save(path_to_npy, df.to_numpy())
        df.to_csv(path_to_csv, index=False)
        
def save_user_test_data(path_to_test, path_to_npy="data/WACA/WACA/WACA_test.npy", path_to_csv="data/WACA/WACA/test.csv"):
    """
    path_to_test :  filepath to the user data you are testing against
    path_to_csv :   name of the file to save the processed test data
    

    """
    df = clean_data(path_to_test)
    if has_sufficient_data(df):
        np.save(path_to_npy, df.to_numpy())
        df.to_csv(path_to_csv, index=False)        

def save_sensor_specific_train_data(path_to_train, sensor_type, path_to_npy="data/WACA/WACA/WACA_train_sensor.npy", path_to_csv="data/WACA/WACA/train_sensor.csv"):
    """
    FOR GYROSCROPE: sensor_type = 10
    FOR ACCELEROMETER: sensor_type = 4    
    """
    
    df = clean_data(path_to_train)
    if df is not None:
        df_sensor = df[df['sensor'] == sensor_type]
        if not df_sensor.empty:
            np.save(path_to_npy, df_sensor.to_numpy())
            df_sensor.to_csv(path_to_csv, index=False)
            
def save_sensor_specific_test_data(path_to_test, sensor_type, 
                                   path_to_npy="data/WACA/WACA/WACA_test_sensor.npy", 
                                   path_to_csv="data/WACA/WACA/test_sensor.csv"):
    """
    Save test data specific to a given sensor type.

    path_to_test :  filepath to the user data you are testing against
    sensor_type  :  sensor type to filter (e.g., 10 for gyroscope, 4 for accelerometer)
    path_to_npy  :  Name of the npy file to save
    path_to_csv  :  Name of the csv file to save
    """
    
    df = clean_data(path_to_test)
    if df is not None:
        df_sensor = df[df['sensor'] == sensor_type]
        if not df_sensor.empty:
            np.save(path_to_npy, df_sensor.to_numpy())
            df_sensor.to_csv(path_to_csv, index=False)

            
        
def save_user_test_label_data(impostor, 
                              path_to_csv="data/WACA/WACA/test.csv", 
                              path_to_test_label="data/WACA/WACA/test_label.csv"):
    """
    Save the testing data and create a labels file.

    impostor    :   True if the user is an impostor, False if the user is the authentic user       
    path_to_csv :   Name of the csv file containing the test data
    path_to_test_label : Name of the test label csv file to save
    """

    # Assumes the user test data has already been saved using save_user_test_data

    # Load the saved test data to create labels
    df = pd.read_csv(path_to_csv)
    
    # Create labels based on impostor status
    labels = [1 if impostor else 0] * len(df)

    # Create a DataFrame for labels
    labels_df = pd.DataFrame({
        "timestamp": df["timestamp"],
        "label": labels
    })

    print(labels_df.head())

    # Save the labels DataFrame to a CSV file
    labels_df.to_csv(path_to_test_label, index=False)

        

if __name__ == "__main__":
    path_to_train = "data/WACA/WACA/WACA_dataset/user1_2.csv"
    save_user_train_data(path_to_train)

    # Load the saved .npy file
    loaded_data = np.load("data/WACA/WACA/WACA_train.npy")
    print(len(loaded_data))