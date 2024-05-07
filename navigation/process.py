# txt to csv
import pandas as pd
from tqdm import tqdm
import csv
import os

from tools import *
from earth import lla2ned, ned2lla
data_path = './KF-GINS/dataset'
#  names=['week', 'seconds', 'lat', 'lon', 'h', 'vn', 've', 'vd', 'roll','pitch','yaw']

# truth.nav 转 gt.txt
ref_pos_lla = np.array([[30.4447873701], [114.4718632047], [20.899]])

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


# KF_GINS_Navresult.nav 转 kf_gins.txt
with open(os.path.join(data_path,'KF_GINS_Navresult.nav'), mode='r', encoding='utf-8') as f:
    lines = f.readlines()

with open(os.path.join(data_path, 'fused_kfgins.txt'), 'a') as save_file:
    for iter, line in enumerate(tqdm(lines, desc="Processing lines")):
        numbers = [float(num) for num in line.split()]
        time = numbers[1]
        pos_lla = np.array(numbers[2:5]).reshape(3, 1)
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
