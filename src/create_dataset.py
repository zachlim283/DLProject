import os
import math
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def generate_datasets(RAW_DATASET_DIR=os.path.join(".", "input", "PHYTMO")):

    def check_params(start_time: int, target_sample_rate: int, window_size: int, interval: float, controller_side: str):
        if (start_time < 0) or (start_time > 20):
            raise Exception("Check start_time!")
        
        if (target_sample_rate < 1) or (target_sample_rate > 333):
            raise Exception("Check target_sample_rate!")
        
        if window_size < 1:
            raise Exception("Check window_size!")
        
        if interval < 0:
            raise Exception("Check interval!")
        
        for c in controller_side:
            if c not in ["Larm", "Lforearm", "Rarm", "Rforearm"]:
                raise Exception("Check controller_side!")


    def preprocess_data(data: np.array, input_rate: int, output_rate: int, start_time: int, end_time: int):
        # Drop first few seconds of data
        n_start = None if int(input_rate * start_time) == 0 else int(input_rate * start_time)
        n_end = None if -int(input_rate * end_time) == 0 else -int(input_rate * end_time)
        data = data[n_start:n_end]

        # Downsample data rate
        downsampled_data = []
        ratio = math.ceil(input_rate/output_rate)
        print(f'{ratio=}')
        for i in range(0, len(data)-ratio, ratio):
            downsampled_data.append(np.mean(data[i : i + ratio], axis=0))
        data = np.array(downsampled_data)
        data = data[:, 1:]

        # Minmax Scaling
        scaler = preprocessing.MinMaxScaler()
        data = scaler.fit_transform(data)

        return data


    def create_windowed_timeseries(raw_timeseries: np.array, data_rate: int, window_size: int, interval: float):
        n_window = int(data_rate * window_size)
        n_interval = int(data_rate * interval)

        windowed_timeseries = []

        for i in range(0, len(raw_timeseries) - n_window, n_interval):
            windowed_timeseries.append(raw_timeseries[i : i + n_window])

        return np.array(windowed_timeseries)


    def create_train_test_data(dataset_dir: os.path, input_rate: int, output_rate: int, start_time: int, end_time: int, window_size: int, interval: int, controller_side: str):
        train_X = None
        train_y = None

        for age_group in os.listdir(dataset_dir):
            # traverse through each age group folder to obtain data
            data_directory = os.path.join(dataset_dir, age_group, controller_side)
            
            for file_name in os.listdir(data_directory):
                action_name = file_name[3:6]
                if action_name in ACTIONS_UPPERLIMBS.keys():
                    temp_x = np.genfromtxt(os.path.join(data_directory, file_name), skip_header=True, dtype=float, delimiter=',')

                    # Preprocess Data
                    temp_x = preprocess_data(temp_x, input_rate, output_rate, start_time, end_time)

                    # Create Windowed Timeseries
                    temp_x = create_windowed_timeseries(temp_x, output_rate, window_size, interval)
                    
                    # Create Corresponding Labels - consider if exercise is done correctly (0: correct, 1: wrong)
                    is_correct = file_name[6]
                    if (is_correct == '0'):
                        temp_y = np.array([ACTIONS_UPPERLIMBS[action_name]] * temp_x.shape[0])
                    else:
                        temp_y = np.array([ACTIONS_UPPERLIMBS["NONE"]] * temp_x.shape[0])
                        # temp_y = np.array([ACTIONS_UPPERLIMBS[f"N_{action_name}"]] * temp_x.shape[0])
                    
                    if train_X is None:
                        train_X = temp_x
                        train_y = temp_y
                    
                    else:
                        train_X = np.concatenate((train_X, temp_x))
                        train_y = np.concatenate((train_y, temp_y))

                    print(f'Completed {file_name}, {train_X.shape=}, {train_y.shape=}')
        
        print("Splitting dataset into train-test-val of 80-20-10 ratio...")
        # Split the data and labels into training and temporary set
        X_train_val, X_test, Y_train_val, Y_test = train_test_split(train_X, train_y, test_size=0.2, random_state=42)
        # Split the temporary set into training and validation
        X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=0.125, random_state=42)
        
        return X_train, Y_train, X_test, Y_test, X_val, Y_val


    def save_data(data: np.array, filename: str):
        np.save(filename, data)
        print(f'File saved as {filename}!')


    ACTIONS_UPPERLIMBS = { 'EFE': 0, 'EAH': 1, 'SQZ': 2, 'NONE': 3 }
    UPPERLIMB_DATASET_DIR = os.path.join(RAW_DATASET_DIR, "inertial", "upper")
    GEN_DATASET_DIR = os.path.join(".", "input")
    RAW_DATA_RATE = 100     # 100 Hz for the gyroscopes and accelerometers and to 20 Hz for the magnetometers

    ############################################################################
    ##########################       PARAMETERS       ##########################
    ############################################################################

    start_time = 3                      # Seconds to ignore at the start of each file
    end_time = 5                        # Seconds to ignore at the end of each file
    target_sample_rate = 20             # Downsample data to this rate (Hz)
    window_size = 3                     # Length of window in seconds
    interval = 0.5                      # Time between windows in seconds
    controller_side = ['Larm', 'Rarm']  # Create dataset for Larm/Lforearm/Rarm/Rforearm

    ############################################################################
    ############################################################################
    ############################################################################

    # Check Params
    check_params(start_time, target_sample_rate, window_size, interval, controller_side)
    
    for c in controller_side:
        dataset_ver = f'{window_size}w{str(interval).replace(".", "")}s_{c}'

        # Import data from CSV and create Windowed Timeseries
        train_data, train_labels, test_data, test_labels, val_data, val_labels = create_train_test_data(dataset_dir=UPPERLIMB_DATASET_DIR,
                                                                                    input_rate=RAW_DATA_RATE,
                                                                                    output_rate=target_sample_rate,
                                                                                    start_time=start_time,
                                                                                    end_time=end_time, 
                                                                                    window_size=window_size,
                                                                                    interval=interval, 
                                                                                    controller_side=c)
        print(f'Train Set, {train_data.shape=}, {train_labels.shape=}')
        print(f'Test Set, {test_data.shape=}, {test_labels.shape=}')
        print(f'Val Set, {val_data.shape=}, {val_labels.shape=}')
        
        # Save to root dataset folder for active use
        save_data(train_data, os.path.join(GEN_DATASET_DIR, f'train_data_{dataset_ver}'))
        save_data(train_labels, os.path.join(GEN_DATASET_DIR, f'train_labels_{dataset_ver}'))
        save_data(test_data, os.path.join(GEN_DATASET_DIR, f'test_data_{dataset_ver}'))
        save_data(test_labels, os.path.join(GEN_DATASET_DIR, f'test_labels_{dataset_ver}'))        
        save_data(val_data, os.path.join(GEN_DATASET_DIR, f'val_data_{dataset_ver}'))
        save_data(val_labels, os.path.join(GEN_DATASET_DIR, f'val_labels_{dataset_ver}'))
        





