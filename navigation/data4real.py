import numpy as np 
import pandas as pd
import queue
import os
from tqdm import tqdm

class IMUData(object):
    def __init__(self, time, imu_angle_increment, imu_vel_increment) -> None:
        self.imu_time = time
        self.imu_angle_increment = imu_angle_increment # [rad]
        self.imu_vel_increment = imu_vel_increment

    @staticmethod
    def read_imu_data(data_file)->queue.Queue:
        imu_data_queue = queue.Queue()
        with open(data_file, mode='r', encoding='utf-8') as f:
            lines = f.readlines()
            for iter, line in enumerate(tqdm(lines, desc="reading imu data")):
                numbers = [float(num) for num in line.split()]
                time = numbers[0]
                imu_angle_increment = np.array(numbers[1:4]).reshape(3, 1)
                imu_vel_increment = np.array(numbers[4:7]).reshape(3, 1)
                imu_data = IMUData(time, imu_angle_increment, imu_vel_increment)
                imu_data_queue.put(imu_data)
        print('read imu data total: ', imu_data_queue.qsize())
        return imu_data_queue
    
    @staticmethod
    def Interpolate(last_imu, curr_imu, time):
        # 内插得到time时刻的imd_imu，更新curr_imu
        ratio = (time - last_imu.imu_time) / (curr_imu.imu_time - last_imu.imu_time)
        mid_imu = IMUData(time, ratio * curr_imu.imu_angle_increment, ratio * curr_imu.imu_vel_increment)
        curr_imu.imu_angle_increment = curr_imu.imu_angle_increment - mid_imu.imu_angle_increment
        curr_imu.imu_vel_increment = curr_imu.imu_vel_increment - mid_imu.imu_vel_increment
        return mid_imu

class GNSSData(object):
    def __init__(self, time, position_lla, gnss_std) -> None:
        self.gnss_time = time
        self.position_lla = position_lla
        self.gnss_std = gnss_std
    @staticmethod
    def read_gnss_data(data_file)->queue.Queue:
        gnss_data_queue = queue.Queue()
        with open(data_file, mode='r', encoding='utf-8') as f:
            lines = f.readlines()
            for iter, line in enumerate(tqdm(lines, desc="reading gnss data")):
                numbers = [float(num) for num in line.split()]
                time = numbers[0]
                pos_lla = np.array(numbers[1:4]).reshape(3, 1)
                gnss_std = np.array(numbers[4:7]).reshape(3, 1)
                gnss_data = GNSSData(time, pos_lla, gnss_std)
                gnss_data_queue.put(gnss_data)
        print('read gnss data total: ', gnss_data_queue.qsize())
        return gnss_data_queue
    
class ODOData(object):
    def __init__(self, time, v_b, v_n) -> None:
        self.odo_time = time
        self.v_b = v_b
        self.v_n = v_n
    
    @staticmethod
    def read_odo_data(data_file)->queue.Queue:
        odo_data_queue = queue.Queue()
        with open(data_file, mode='r', encoding='utf-8') as f:
            lines = f.readlines()
            for iter, line in enumerate(tqdm(lines, desc="reading odo data")):
                numbers = [float(num) for num in line.split()]
                time = numbers[0]
                v_b = np.array(numbers[1:4]).reshape(3, 1)
                v_n = np.array(numbers[4:7]).reshape(3, 1)
                odo_data = ODOData(time, v_b, v_n)
                odo_data_queue.put(odo_data)
        print('read odo data total: ', odo_data_queue.qsize())
        return odo_data_queue
    
if __name__ == "__main__":
    imu1 = IMUData(1, np.array([1, 2, 3]).reshape(3, 1), np.array([4, 5, 6]).reshape(3, 1))
    imu2 = IMUData(2, np.array([2, 3, 4]).reshape(3, 1), np.array([5, 6, 7]).reshape(3, 1))
    imu3 = IMUData.Interpolate(imu1, imu2, 1.5)
    print(imu2.imu_angle_increment)