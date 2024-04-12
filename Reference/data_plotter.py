import os
import csv
import numpy as np
import matplotlib.pyplot as plt


def load_data(fp: str, header=True):
    with open(fp, 'r') as f:
        reader = csv.reader(f)
        raw_data = list(reader)

        if header:
            raw_data.pop(0)

        data = np.array(raw_data)

    return data


if __name__ == "__main__":
    ROOT_DIR = os.path.join('.', 'PHYTMO/inertial/upper')
    ACTIONS_UPPERLIMBS = { 'EFE': 0, 'EAH': 1, 'SQZ': 2, 'NONE': 3 }
    controller_side = 'Larm'

    for age_group in os.listdir(ROOT_DIR):
        # traverse through each age group folder to obtain data
        data_directory = os.path.join(ROOT_DIR, age_group, controller_side)
        
        for file_name in os.listdir(data_directory):
            action_name = file_name[3:6]
            if action_name in ACTIONS_UPPERLIMBS.keys():
                f_path = os.path.join(data_directory, file_name)

                IMUdata = load_data(f_path)  
                start_idx = 0
                end_idx = None

                # Extract Data
                gyr = IMUdata[start_idx:end_idx, (1, 2, 3)].astype('float64')
                acc = IMUdata[start_idx:end_idx, (4, 5, 6)].astype('float64')
                quat = IMUdata[start_idx:end_idx, (7, 8, 9)].astype('float64')

                # Plot
                fig, (ax1, ax2, ax3) = plt.subplots(3)
                fig.suptitle(f'Recorded Raw Data {file_name}')
                
                ax1.plot(gyr)
                ax1.set_title('Gyro Data')

                ax2.plot(acc)
                ax2.set_title('Accel Data')
                
                ax3.plot(quat)
                ax3.set_title('Mag Data')

                plt.show()
