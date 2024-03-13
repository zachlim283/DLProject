import socket
import utils
import numpy as np
from math import floor
from time import perf_counter


def downsample(data, output_samples):
    input_len = data.shape[0]
    ratio = floor(input_len/output_samples)

    data_out = []
    for i in range(0, len(data)-ratio, ratio):
        data_out.append(np.mean(data[i : i + ratio], axis=0))

    return np.array(data_out)


CLIENT_IPS = ['192.168.1.25']    # Controller IPs
UDP_IP = '192.168.1.14'         # Host Computer IP
UDP_PORT = 5005

WINDOW_LEN = 3
INTERVAL_LEN = 0.5

sock = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP
sock.bind((UDP_IP, UDP_PORT))

for CLIENT_IP in CLIENT_IPS:
    udp_set = utils.set_udp(CLIENT_IP, UDP_IP, UDP_PORT)

window_data = []
window_data_downsampled = []
window_start = perf_counter()
while (window_time := (perf_counter() - window_start)) < WINDOW_LEN:

    int_data = []
    int_start = perf_counter()
    while (int_time := (perf_counter() - int_start)) < INTERVAL_LEN:
        data_raw, addr = sock.recvfrom(1024) # buffer size is 1024 bytes
        data = data_raw.decode('utf-8')
        int_data.append(data_raw.decode('utf-8').split(',')[2:])
    
    downsample_data = downsample(np.array(int_data, dtype=float), 10)
    print(f'Original: {len(int_data)} samples in {int_time:.4f}s. Data Rate = {(len(int_data)/int_time):.3f} Hz')
    print(f'Downsampled: {downsample_data.shape[0]} samples in {int_time:.4f}s. Data Rate = {(downsample_data.shape[0]/int_time):.3f} Hz')
    window_data.append(int_data)
    window_data_downsampled.append(downsample_data)

window_data_arr = np.concatenate(window_data)
window_data_downsampled_arr = np.concatenate(window_data_downsampled)
print(f'Original: {window_data_arr.shape[0]} samples in {window_time:.4f}s. Avg Data Rate = {(window_data_arr.shape[0]/window_time):.3f} Hz')
print(f'Downsampled: {window_data_downsampled_arr.shape[0]} samples in {window_time:.4f}s. Avg Data Rate = {(window_data_downsampled_arr.shape[0]/window_time):.3f} Hz')
