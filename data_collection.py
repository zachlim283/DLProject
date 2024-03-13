import os
import csv
import socket
import utils
from time import time, sleep


device_names = {'30:AE:A4:4D:69:C0': 'Test Device',
                '84:FC:E6:00:B4:94': 'XIAO_ESP_1',
                '84:FC:E6:00:AF:08': 'XIAO_ESP_2',
                '84:FC:E6:00:B9:E4': 'DEV_BOARD',
                '84:FC:E6:00:C1:60': 'BLUE_RH',
                '48:31:B7:3F:DD:BC': 'BLUE_LH',
                '84:FC:E6:00:A6:B8': 'GREEN_RH',
                '84:FC:E6:00:AF:08': 'GREEN_LH'}


def increment_filename(output_filename):
    no_ext = output_filename.split('.')[0]
    last_3 = int(no_ext[-3:])
    last_3 += 1
    last_3 = '0' * (3 - len(str(last_3))) + str(last_3)
    output_filename_new =no_ext[:-3] + last_3 + '.csv'

    return output_filename_new

def save_data(output_filename: str, data: list):
    header = ['gx', 'gy', 'gz', 'ax', 'ay', 'az', 'qw', 'qx', 'qy', 'qz']

    with open(output_filename, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)


if __name__ == '__main__':
    
    ###########################################################################################################
    ####################################           CONFIGURATION           ####################################
    ###########################################################################################################
    
    # Go to 192.168.254, Click 'Connected Devices' in the left panel. Update the following:

    CLIENT_IPS = ['192.168.1.25' ,'192.168.1.24']       # Controller IPs
    UDP_IP = '192.168.1.16'                              # Host Computer IP

    MOVEMENT_NAME = 'Testing'                           # Running, Swimming, Kayaking, Longkang Fishing, Reeling

    ###########################################################################################################
    ###########################################################################################################

    UDP_PORT = 5005
    WINDOW_SIZE = 60

    root_folder = 'Recorded_Movement_Data'

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))

    recorded_data_all = []
    recorded_data_split = {}

    print('Connecting to and Calibrating Devices! Please hold still...')
    sleep(5)
    for CLIENT_IP in CLIENT_IPS:
        try:
            recorded_data_split[utils.get_MAC(CLIENT_IP)] = []
            utils.calibrate_sensor(CLIENT_IP)
            utils.set_udp(CLIENT_IP, UDP_IP, UDP_PORT)
            
        except:
            print('An Error Occurred! Check IPs and ensure devices are on')
            raise SystemExit
        
    print(f'Calibration Complete! Devices connected: {[dev for dev in recorded_data_split.keys()]}')
    print('Get ready for data recording in 3...')
    sleep(1)
    print('2...')
    sleep(1)
    print('1...')
    sleep(1)

    print('Recording Started!')
    start = time()
    while time()-start < WINDOW_SIZE:
        data_raw, addr = sock.recvfrom(1024) # buffer size is 1024 bytes
        data = data_raw.decode('utf-8')
        recorded_data_all.append(data.split(','))

    print(f'Data Collected! Parsing...')

    for datapoint in recorded_data_all:
        device = datapoint[0]
        recorded_data_split[device].append(datapoint[2:])

    for dev, data in recorded_data_split.items():
        filename = os.path.join(root_folder, MOVEMENT_NAME, f'{device_names[dev]}_001.csv')
        
        while os.path.exists(filename):
            filename = increment_filename(filename)

        save_data(filename, data)
        print(f'Data Saved to {filename}!')
        print(f'Device: {device_names[dev]} \tSamples: {len(data)} \tData Rate: {len(data)/WINDOW_SIZE}')
