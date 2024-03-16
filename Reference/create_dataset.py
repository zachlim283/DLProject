import os
import math
import numpy as np
from datetime import datetime
from sklearn import preprocessing


def check_params(start_time: int, target_sample_rate: int, window_size: int, interval: float, controller_side: str):
    if (start_time < 0) or (start_time > 20):
        raise Exception("Check start_time!")
    
    if (target_sample_rate < 1) or (target_sample_rate > 333):
        raise Exception("Check target_sample_rate!")
    
    if window_size < 1:
        raise Exception("Check window_size!")
    
    if interval < 0:
        raise Exception("Check interval!")
    
    if controller_side not in ["RH", "LH", "BH"]:
        raise Exception("Check controller_side!")


def preprocess_data(data: np.array, input_rate: int, output_rate: int, start_time: int):
    
    # Drop first few seconds of data
    n_start = int(input_rate * start_time)
    data = data[n_start:]

    # Downsample data rate
    downsampled_data = []
    ratio = math.ceil(input_rate/output_rate)
    for i in range(0, len(data)-ratio, ratio):
        downsampled_data.append(np.mean(data[i : i + ratio], axis=0))
    data = np.array(downsampled_data)

    # Remove Outliers
    # TODO

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


def create_train_data(dataset_dir: os.path, input_rate: int, output_rate: int, start_time: int, window_size: int, interval: int, controller_side: str):
    if controller_side == 'BH':
        controller_side = 'H'

    train_X = None
    train_y = None

    for action_name in os.listdir(dataset_dir):
        if action_name in ACTIONS.keys():
            for data_csv in os.listdir(os.path.join(dataset_dir, action_name)):
                if controller_side in data_csv:
                    temp_x = np.genfromtxt(os.path.join(dataset_dir, action_name, data_csv), skip_header=True, dtype=float, delimiter=',')

                    # Preprocess Data
                    temp_x = preprocess_data(temp_x, input_rate, output_rate, start_time)

                    # Create Windowed Timeseries
                    temp_x = create_windowed_timeseries(temp_x, output_rate, window_size, interval)
                    
                    # Create Corresponding Labels
                    temp_y = np.array([ACTIONS[action_name]] * temp_x.shape[0])                

                    if train_X is None:
                        train_X = temp_x
                        train_y = temp_y
                    
                    else:
                        train_X = np.concatenate((train_X, temp_x))
                        train_y = np.concatenate((train_y, temp_y))

            print(f'Completed {action_name}, {train_X.shape=}, {train_y.shape=}')
    
    return train_X, train_y


def create_test_data(dataset_dir: os.path, input_rate: int, output_rate: int, start_time: int, window_size: int, interval: int):

    test_X_RH = None
    test_X_LH = None

    for data_csv in os.listdir(os.path.join(dataset_dir, 'Testing')):
        if data_csv.split('.')[1] != 'csv':
            continue
        
        temp = np.genfromtxt(os.path.join(dataset_dir, 'Testing', data_csv), skip_header=True, dtype=float, delimiter=',')

        # Preprocess Data
        temp = preprocess_data(temp, input_rate, output_rate, start_time)

        # Create Windowed Timeseries
        temp = create_windowed_timeseries(temp, output_rate, window_size, interval)

        if 'RH' in data_csv:               
            if test_X_RH is None:
                test_X_RH = temp

            else:
                test_X_RH = np.concatenate((test_X_RH, temp))

        elif 'LH' in data_csv:
            if test_X_LH is None:
                test_X_LH = temp

            else:
                test_X_LH = np.concatenate((test_X_LH, temp))
        

    print(f'Test Data Generated! {test_X_RH.shape=}, {test_X_LH.shape=}')
    
    return test_X_RH, test_X_LH



def save_data(data: np.array, filename: str):
    np.save(filename, data)
    print(f'File saved as {filename}!')


if __name__ == '__main__':
    ACTIONS_ALL = {'Breaststroke': 0, 'Flying': 1, 'Freestyle': 2, 'Kayaking': 3, 'LeftReeling': 4, 'RightReeling': 5, 'Rowing': 6, 'Running': 7, 'Shopping': 8}
    ACTIONS = {'Kayaking': 0, 'LeftReeling': 1, 'RightReeling': 2, 'Running': 3, 'Shopping': 4}
    RAW_DATASET_DIR = os.path.join('.', 'Recorded_Movement_Data')
    GEN_DATASET_DIR = os.path.join('.', 'Generated_Datasets')
    RAW_DATA_RATE = 333

    ############################################################################
    ##########################       PARAMETERS       ##########################
    ############################################################################

    start_time = 3                  # Seconds to ignore at the start of each file
    target_sample_rate = 20         # Downsample data to this rate (Hz)
    window_size = 3                 # Length of window in seconds
    interval = 0.5                  # Time between windows in seconds
    controller_side = 'BH'          # Create dataset for RH/LH/BH
    create_test = True              # Generate Test Data (Always BH)

    ############################################################################
    ############################################################################
    ############################################################################

    # Check Params
    check_params(start_time, target_sample_rate, window_size, interval, controller_side)
    
    # Remove current dataset
    for entry in os.listdir(GEN_DATASET_DIR):
        if entry.split('.')[-1] == 'npy':
            os.remove(os.path.join(GEN_DATASET_DIR, entry))

    # Create dated directory for new dataset
    dataset_ver = f'{window_size}w{str(interval).replace(".", "")}s_{controller_side}'
    date = datetime.now().strftime("%y%m%d_%H%M")

    save_folder = os.path.join(GEN_DATASET_DIR, date)
    os.mkdir(save_folder)

    # Import data from CSV and create Windowed Timeseries
    train_data, train_labels = create_train_data(dataset_dir=RAW_DATASET_DIR,
                                                 input_rate=RAW_DATA_RATE,
                                                 output_rate=target_sample_rate,
                                                 start_time=start_time, 
                                                 window_size=window_size,
                                                 interval=interval, 
                                                 controller_side=controller_side)

    # Save data to dated folder for records
    save_data(train_data, os.path.join(save_folder, f'train_data_{dataset_ver}'))
    save_data(train_labels, os.path.join(save_folder, f'train_labels_{dataset_ver}'))

    # Save to root dataset folder for active use
    save_data(train_data, os.path.join(GEN_DATASET_DIR, f'train_data_{dataset_ver}'))
    save_data(train_labels, os.path.join(GEN_DATASET_DIR, f'train_labels_{dataset_ver}'))


    # Create Test Set if required
    if create_test:
        print('Generating Test Data...')
        test_data_RH, test_data_LH = create_test_data(dataset_dir=RAW_DATASET_DIR,
                                                      input_rate=RAW_DATA_RATE,
                                                      output_rate=target_sample_rate,
                                                      start_time=start_time,
                                                      window_size=window_size,
                                                      interval=interval)
        
        save_data(test_data_RH, os.path.join(save_folder, f'test_data_RH_{dataset_ver}'))
        save_data(test_data_LH, os.path.join(save_folder, f'test_data_LH_{dataset_ver}'))
        save_data(test_data_RH, os.path.join(GEN_DATASET_DIR, f'test_data_RH_{dataset_ver}'))
        save_data(test_data_LH, os.path.join(GEN_DATASET_DIR, f'test_data_LH_{dataset_ver}'))
        

    '''
    Train Data Shape: (n, 60, 10)
        n:  No. of samples
        60: 3s window of data after downsampling
        10: Values from MPU (gx,gy,gz,ax,ay,az,qw,qx,qy,qz)
    '''




