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
    IMUdata = load_data('PHYTMO/inertial/upper/A/Larm/A01EAH1_2.csv')  
    start_idx = 700
    end_idx = -1000

    # Extract Data
    gyr = IMUdata[start_idx:end_idx, (1, 2, 3)].astype('float64')
    acc = IMUdata[start_idx:end_idx, (4, 5, 6)].astype('float64')
    quat = IMUdata[start_idx:end_idx, (7, 8, 9)].astype('float64')

    # Plot
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    fig.suptitle('Recorded Raw Data')
    
    ax1.plot(gyr)
    ax1.set_title('Gyro Data')

    ax2.plot(acc)
    ax2.set_title('Accel Data')
    
    ax3.plot(quat)
    ax3.set_title('Mag Data')

    plt.show()
