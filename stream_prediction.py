import socket
import requests
import pickle
import torch
import resampy
import numpy as np
from math import floor
from utils import set_udp, calibrate_sensor
from time import sleep, perf_counter
from train_encoder import Encoder
from sklearn import preprocessing
from collections import deque


def downsample(data: np.array, output_samples: int) -> np.array:
    input_len = data.shape[0]
    ratio = floor(input_len/output_samples)

    data_out = []
    for i in range(output_samples):
        data_out.append(np.mean(data[i : i + ratio], axis=0))

    return np.array(data_out)


def stream_predict(CLIENT_IP: str, UDP_IP: str, UDP_PORT: int, 
                   tgt_sample_rate: int, window_size: int, interval: int,
                   encoder_model, classifier, 
                   thread_num: int) -> None:
    
    print(f'Thread {thread_num} started for {CLIENT_IP}...')
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))

    udp_set = set_udp(CLIENT_IP, UDP_IP, UDP_PORT)

    if udp_set:
        print(f'Controller at {CLIENT_IP} connected!')

    print('Calibrating Sensor...')
    sleep(2)
    calibrate_sensor(CLIENT_IP)
    
    # Continuously collect data
    # Every {interval=0.5}s, output {tgt_sample_rate * interval=10} data points, append to deque
    # Deque is sized for {window_size / interval=6} sets of 10 datapoints
    samples_per_interval = int(tgt_sample_rate * interval)      # 20 * 0.5 = 10
    intervals_per_window = window_size / interval               # 3 / 0.5 = 6
    
    data_window = deque(maxlen=int(intervals_per_window))

    while True:
        int_start = perf_counter()
        data_stream = []

        while perf_counter() - int_start < interval:
            data_raw, addr = sock.recvfrom(1024) # buffer size is 1024 bytes
            data_stream.append(data_raw.decode('utf-8').split(',')[2:])

        interval_downsampled = downsample(np.array(data_stream, dtype=float), samples_per_interval)
        data_window.append(interval_downsampled)

        if len(data_window) < 6:        # Ensure data window is full (6 sets of 10 x 10 arrays)
            continue

        data_window = np.concatenate(data_window)
        scaled_data = preprocessing.MinMaxScaler().fit_transform(data_window)
        scaled_data = torch.from_numpy(scaled_data.reshape(1, 60, 10)).type(torch.float32).to(device)
        encoded_data = encoder_model(scaled_data).cpu()
        encoded_data = encoded_data.detach().numpy()
        prediction = classifier.predict(encoded_data)
        print(prediction)
        # print(f'Prediction = {ACTIONS[prediction[0]]}')

    # data_rate = 333

    # n_downsample_window = 17    # ceil(data_rate/tgt_sample_rate)
    # n_interval = 10             # int(tgt_sample_rate * interval)
    # n_model_window = 60         # int(tgt_sample_rate * window_size)
    
    # data_window = deque(maxlen=n_model_window)
    # interval_cnt = -n_model_window
    
    # print('Starting predictions...')
    # while True:
    #     data_downsample = np.zeros(shape=(n_downsample_window, 10), dtype=float)

    #     start = perf_counter()
    #     for i in range(n_downsample_window):
    #         data_raw, addr = sock.recvfrom(1024) # buffer size is 1024 bytes
    #         data_downsample[i] = data_raw.decode('utf-8').split(',')[2:]
    #     stop = perf_counter()

    #     data_window.append(np.mean(data_downsample, axis=0))

    #     interval_cnt += 1
    #     if interval_cnt > n_interval:
    #         scaled_data = preprocessing.MinMaxScaler().fit_transform(np.array(data_window))
    #         encoded_data = encoder_model.predict(scaled_data.reshape(1, 60, 10))
    #         prediction = classifier.predict(encoded_data)
    #         print(f'Prediction = {ACTIONS[prediction[0]]}')
    #         interval_cnt = 0

    #     data_rate = n_downsample_window/(stop-start)


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device("mps") 
    else:
        device = torch.device("cpu")
    print(device)

    CLIENT_IPS = ['192.168.1.25']                       # Controller IPs
    UDP_IP = '192.168.1.14'                             # Host Computer IP
    UDP_PORT_START = 5000
    ACTIONS = {'Kayaking': 0, 'LeftReeling': 1, 'RightReeling': 2, 'Running': 3, 'Shopping': 4}

    window_size = 3
    interval = 0.5
    dataset_ver = '3w05s_BH'
    encoder_ver = f'360h20e'

    # TF Model
    # encoder_model = load_model('models/tanh_360_20_16/3w05s_Both_encoder_ag.h5')

    # PyTorch Model
    # encoder_model = torch.load(f'ML_Models/{dataset_ver}_{encoder_ver}_encoder.pt').to(device)
    encoder_model = torch.load(f'ML_Models/3w05s_BH_encoder.pt', map_location=device).to(device)
    encoder_model.eval()
    print('Encoder Model Loaded!')

    # svm_file = open('models/tanh_360_20_16/3w05s_Both_svm_classifier_encoder_ag.pkl', 'rb')
    svm_file = open('ML_Models/3w05s_BH_svm_classifier.pkl', 'rb')
    svm_classifier = pickle.load(svm_file)
    print('Classifier Loaded!')

    i = 0
    stream_predict(CLIENT_IPS[i], UDP_IP, UDP_PORT_START+i,
                   tgt_sample_rate=20, window_size=window_size, interval=interval, 
                   encoder_model=encoder_model, classifier=svm_classifier, 
                   thread_num=i)

    # THREAD_POOL = ThreadPoolExecutor(len(CLIENT_IPS))

    # for i in range(len(CLIENT_IPS)):
        # THREAD_POOL.submit(stream_predict,
        #                    CLIENT_IPS[i], UDP_IP, UDP_PORT_START+i,
        #                    tgt_sample_rate=20, thread_num=i, 
        #                    window_size=window_size, interval=interval, 
        #                    model=svm_classifier)