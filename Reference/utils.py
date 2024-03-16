import requests
import json
import csv
import socket
import subprocess
from itertools import chain
from concurrent.futures import ThreadPoolExecutor


# def scan_network(use_cache: bool):
#     cache_file = 'dev_cache'
#     if use_cache == 1:
#         device_list = read_cache(cache_file)
#         new_device_list = []

#         for dev in device_list:
#             URL = 'http://' + dev['IP'] + '/get_MAC'

#             try:
#                 response = requests.get(URL, timeout=(3, 5))
#                 if response.status_code == 200:
#                     dev['MAC'] = response.json()[0]
#                     new_device_list.append(dev)
            
#             except requests.exceptions.Timeout:
#                 print('Request timed out!')
#                 continue

#             except requests.exceptions.RequestException as e:
#                 print('Device is unrecognised, IP may have changed.')
#                 continue
        
#         write_cache(cache_file, new_device_list)


#     else:
#         device_list = []
#         for i in range(1, 30):
#             IP = '192.168.1.' + str(i)
#             URL = 'http://' + IP + '/get_MAC'
#             print('Trying ' + URL + '...')

#             try:
#                 response = requests.get(URL, timeout=(0.2, 0.2))
#                 if response.status_code == 200:
#                     print('Device Found!')
#                     MAC = response.json()[0]
#                     device_list.append({'IP': IP, 'MAC': MAC})

#             except requests.exceptions.RequestException as e:
#                 continue
                    
#         write_cache(cache_file, device_list)
    
#     ############## FOR TESTING ################
#     device_list = [{'IP':'192.168.1.9', 'MAC':'30:AE:A4:4D:69:C0'},
#                    {'IP':'192.168.1.20', 'MAC':'84:FC:E6:00:B4:94'},
#                    {'IP':'192.168.1.21', 'MAC':'84:FC:E6:00:AF:08'},
#                    {'IP':'192.168.1.22', 'MAC':'84:FC:E6:00:B9:E4'}]
#     ############## FOR TESTING ################
    
#     return device_list

def find_basestation(MAC: str):
    output = {}
    data = subprocess.check_output(['ipconfig','/all']).decode('utf-8').split('\n')

    for item in data:
        if "Physical Address" in item:
            curr_MAC = item.strip().split(" ")[-1]
            continue

        if "IPv4 Address" in item:
            HOST_IP = item.strip().split(" ")[-1].strip("(Preferred)")
            output[curr_MAC] = {}
            output[curr_MAC]["Host Assigned IP"] = HOST_IP
            continue

        if "Default Gateway" in item:
            GATEWAY_ADD_FULL = item.strip().split(" ")[-1].split(".")
            GATEWAY_ADD = f'{GATEWAY_ADD_FULL[0]}.{GATEWAY_ADD_FULL[1]}.{GATEWAY_ADD_FULL[2]}'
            output[curr_MAC]["Gateway Address"] = GATEWAY_ADD
            continue

    return output.get(MAC)


def scan_network(use_cache: bool, gateway: str, num_threads: int =16):
    cache_file = 'dev_cache'
    if use_cache == 1:
        device_list = read_cache(cache_file)
        new_device_list = []

        for dev in device_list:
            URL = f'http://{dev["IP"]}/get_MAC'

            try:
                response = requests.get(URL, timeout=(3, 5))
                if response.status_code == 200:
                    dev['MAC'] = response.json()[0]
                    new_device_list.append(dev)
            
            except requests.exceptions.Timeout:
                print('Request timed out!')
                continue

            except requests.exceptions.RequestException as e:
                print('Device is unrecognised, IP may have changed.')
                continue
        
        write_cache(cache_file, new_device_list)
        return new_device_list
    
    else:            
        work_results = {}
        thread_pool = ThreadPoolExecutor(num_threads)
        for i in range(num_threads):
            work_results[i] = thread_pool.submit(scan_worker, i, gateway, num_threads)

        device_list = list(chain.from_iterable([x.result() for x in work_results.values()]))
        write_cache(cache_file, device_list)
        thread_pool.shutdown()
        return device_list


def scan_worker(idx: int, gateway: str, num_threads: int):
    start_idx = int(idx * (256/num_threads)) + 1
    partial_dev_list = []
    for i in range(start_idx, start_idx+32):
        IP = f'{gateway}.{i}'
        URL = f'http://{IP}/get_MAC'

        try:
            response = requests.get(URL, timeout=(0.3, 0.3))
            if response.status_code == 200:
                MAC = response.json()[0]
                print(f'Device Found on Thread {idx}: {IP}, {MAC}')
                partial_dev_list.append({'IP': IP, 'MAC': MAC})

        except requests.exceptions.RequestException as e:
            continue
    
    return partial_dev_list


def read_cache(cache_file):
    try:
        with open(cache_file, "rt") as fp:
            data = json.load(fp)

    except IOError:
        print("Could not read file, starting from scratch")
        data = []

    return data


def write_cache(cache_file, data: list):
    with open(cache_file, "wt") as fp:
        json.dump(data, fp)


# def connect_device(IP: str):
#     URL = f'http://{IP}/get_MAC'
#     print('Connecting to ' + IP + '...')
#     try:
#         response = requests.get(URL, timeout=(3, 5))
#         if response.status_code == 200:
#             MAC = response.json()[0]
#             return MAC, "Connected!"
#         else:
#             return None, "An error occurred!"

#     except requests.exceptions.Timeout:
#         return None, "The request timed out"

#     except requests.exceptions.RequestException as e:
#         return None, "Request Exception!"

def connect_device(CLIENT_IP: str, UDP_IP: str, UDP_PORT: int):
    sock = socket.socket(socket.AF_INET,    # Internet
                         socket.SOCK_DGRAM) # UDP
    print(UDP_IP, UDP_PORT)
    sock.bind((UDP_IP, UDP_PORT))

    udp_set = set_udp(CLIENT_IP, UDP_IP, UDP_PORT)

    if udp_set:
        print(f'UDP Port Activated on {CLIENT_IP}...')
        return True, sock
    
    else:
        print(f'Error setting UDP IP on {CLIENT_IP}')
        return False, None


# def disconnect_device(CLIENT_IP: str, sock: socket.socket):
#     try:
#         sock.close()
#         return True
    
#     except:
#         print("Error closing socket!")
#         return False



def set_udp(CLIENT_IP:str, UDP_IP: str, UDP_PORT: int):
    URL = f'http://{CLIENT_IP}/set_UDP?ip={UDP_IP}&port={UDP_PORT}'
    response = requests.get(URL, timeout = (3, 5))

    if response.status_code == 200:
        return True
    
    return False


# def activate_udp(IP: str):
#     URL = f'http://{IP}/start_data'
#     response = requests.get(URL, timeout = (3, 5))
    
#     if response.status_code == 200:
#         return True
    
#     if response.status_code == 409:
#         print("UDP Port not set!")
#         return False

#     return False


# def deactivate_udp(IP: str):
#     URL = f'http://{IP}/stop_data'
#     response = requests.get(URL, timeout = (3, 5))
    
#     if response.status_code == 200:
#         return True
    
#     else: return False 


# def read_MPU(IP: str):
#     URL = 'http://' + IP + '/read_MPU'

#     try:
#         response = requests.get(URL, timeout = (3, 5))
#         data = response.content.decode('utf-8').split(',')

#         # print("Acceleration X: ", data[3], ", Y: ", data[4], " Z: ", data[5], " m/s^2")
#         # print("Rotation X: ", data[0], ", Y: ", data[1], " Z: ", data[2], " rad/s")
#         # print("Temp: ", data[6], " degC")

#         return data
    
#     except requests.exceptions.Timeout:
#         print("The request timed out")
        
#     except requests.exceptions.RequestException as e:
#         print("An error occurred:", e)
    
#     except KeyboardInterrupt:
#         print("User Interrupted!")

def get_MAC(IP: str):
    URL = f'http://{IP}/get_MAC'
    try:
        response = requests.get(URL, timeout = (3, 5))

    except requests.exceptions.Timeout:
        print("The request timed out")

    return response.json()[0]


def calibrate_sensor(IP: str):
    URL = f'http://{IP}/reset'
    try:
        response = requests.get(URL, timeout = (3, 5))
        if response.status_code == 200:
            print('Device Calibrated')
    
    except requests.exceptions.Timeout:
        print("The request timed out")
        
    except requests.exceptions.RequestException as e:
        print("An error occurred:", e)
    
    except KeyboardInterrupt:
        print("User Interrupted!")


def save_data(output_path: str, output_filename: str, data: list):
    header = ['Device MAC', 'Time', 'gx', 'gy', 'gz', 'ax', 'ay', 'az', 'qw', 'qx', 'qy', 'qz']
    
    fp = f'{output_path}/{output_filename}.csv'
    print(fp)

    with open(fp, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)
