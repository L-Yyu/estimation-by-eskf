# txt to csv
import pandas as pd
from tqdm import tqdm
import csv
import os

from tools import *
from earth import lla2ned, ned2lla
data_path = './awesome-gins-datasets/ICM20602'
#  names=['week', 'seconds', 'lat', 'lon', 'h', 'vn', 've', 'vd', 'roll','pitch','yaw']

# truth.nav 转 gt.txt
# ref_pos_lla = np.array([[30.4447873701], [114.4718632047], [20.899]]) # test
ref_pos_lla = np.array([[30.4604298544], [114.4725036426], [23.08011]]) # ICM20602
with open(os.path.join(data_path,'truth.nav'), mode='r', encoding='utf-8') as f:
    lines = f.readlines()

with open(os.path.join(data_path, 'gt.txt'), 'a') as save_file:
    for iter, line in enumerate(tqdm(lines, desc="Processing lines")):
        numbers = [float(num) for num in line.split()]
        time = numbers[1]
        pos_lla = np.array(numbers[2:5]).reshape(3, 1)
        # if iter == 0:
        #     ref_pos_lla = pos_lla
        pos_ned = lla2ned(pos_lla, ref_pos_lla)
        euler = np.array(numbers[8:11])
        quat = euler2quaternion(euler)
        
        # 写入结果到文件
        save_file.write(str(time) + ' ')
        for i in range(3):
            save_file.write(str(pos_ned[i][0]) + ' ')
        for i in range(4):
            if i < 3:
                save_file.write(str(quat[i]) + ' ')
            else:
                save_file.write(str(quat[i]) + '\n')


# truth.nav 转 odom.txt 
with open(os.path.join(data_path,'truth.nav'), mode='r', encoding='utf-8') as f:
    lines = f.readlines()

with open(os.path.join(data_path, 'odo.txt'), 'a') as save_file:
    for iter, line in enumerate(tqdm(lines, desc="Processing lines")):
        numbers = [float(num) for num in line.split()]
        time = numbers[1]
        pos_lla = np.array(numbers[2:5]).reshape(3, 1)
        # if iter == 0:
        #     ref_pos_lla = pos_lla
        v_n = np.array(numbers[5:8]).reshape(3, 1)
        pos_ned = lla2ned(pos_lla, ref_pos_lla)
        euler = np.array(numbers[8:11])
        quat = euler2quaternion(euler)
        rm = euler2rm(euler)
        v_b = rm @ v_n
        
        # 写入结果到文件
        save_file.write(str(time) + ' ')
        for i in range(3):
            save_file.write(str(v_b[i][0]) + ' ')
        for i in range(3):
            save_file.write(str(v_n[i][0]) + ' ')
        save_file.write('\n')


