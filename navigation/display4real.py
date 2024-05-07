import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits import mplot3d
from tools import *

def load_txt_data(data_path):
    try:
        return np.loadtxt(data_path)
    except FileNotFoundError as err:
        print('this is a OSError: ' + str(err))

def evaluate(gt_data, fuse_data):
    rmse_x = np.sqrt(np.mean((fuse_data[:, 1] - gt_data[:, 1])**2))
    rmse_y = np.sqrt(np.mean((fuse_data[:, 2] - gt_data[:, 2])**2))
    rmse_xy = np.sqrt(np.mean((fuse_data[:, 1:3] - gt_data[:, 1:3])**2))
    print('position rmse: ', rmse_x, rmse_y, rmse_xy)
    euler_gt = quaternion2euler(gt_data[:, 4:8])
    euler_fuse = quaternion2euler(fuse_data[:, 4:8])
    d_roll = np.array([angle_diff(gt, fuse) for gt, fuse in zip(euler_gt[:, 0], euler_fuse[:, 0])])
    d_pitch = np.array([angle_diff(gt, fuse) for gt, fuse in zip(euler_gt[:, 1], euler_fuse[:, 1])])
    d_yaw = np.array([angle_diff(gt, fuse) for gt, fuse in zip(euler_gt[:, 2], euler_fuse[:, 2])])
    rmse_roll = np.sqrt(np.mean(d_roll**2))
    rmse_pitch = np.sqrt(np.mean(d_pitch**2))
    rmse_yaw = np.sqrt(np.mean((d_yaw**2)))
    print(euler_gt[-1,:])
    print(euler_fuse[-1,:])
    print('attitude rmse: ', rmse_roll, rmse_pitch, rmse_yaw)

if __name__ == "__main__":
    data = 'test'
    data_path = './data/'+data
    fuse_data_path = data_path+'/fused.txt'
    fuse_15_6_data_path = data_path+'/fused_15_6.txt'
    fuse_15_12_data_path = data_path+'/fused_15_12.txt'
    fuse_21_6_data_path = data_path+'/fused_21_6.txt'
    fuse_21_18_data_path = data_path+'/fused_21_18.txt'
    fuse_odo_data_path = data_path+'/fused_odo.txt'
    gps_data_path = data_path+'/gps_measurement.txt'
    gt_data_path = data_path+'/gt.txt'
    ins_data_path = data_path+'/ins.txt'
    gins_data_path = data_path+'/fused_kfgins.txt'

    data_size = 200000
    gt_start_index = 10052
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_title('real data results comparison')
    if os.path.exists(gt_data_path):
        gt_data = np.loadtxt(gt_data_path)
        # 由于gt_data有冗余数据，需要去除
        print(data_size)
        i = 0
        while i <200200:
            if gt_data[gt_start_index+i+1,0]-gt_data[gt_start_index+i,0]<0.003:
                # print(gt_data[gt_start_index+i+1,0])
                gt_data=np.delete(gt_data,gt_start_index+i+1,axis=0)
                continue
            i += 1
        gt_data = gt_data[gt_start_index:gt_start_index + data_size, :]

        ax.plot3D(gt_data[:, 1], gt_data[:, 2], gt_data[:, 3], color='b', label='ground_truth')
         
    if os.path.exists(fuse_data_path):
        fuse_data = np.loadtxt(fuse_data_path)
        ax.plot3D(fuse_data[:, 1], fuse_data[:, 2], fuse_data[:, 3], color='r', label='fuse_gps_imu')
        # 检查数据是否对齐
        print(fuse_data[0,0],gt_data[0,0])
        print(fuse_data[-1,0],gt_data[-1,0])
        
        print('fuse data:')
        evaluate(gt_data, fuse_data)

    if os.path.exists(fuse_21_6_data_path):
        fuse_21_6_data = np.loadtxt(fuse_21_6_data_path)
        ax.plot3D(fuse_21_6_data[:, 1], fuse_21_6_data[:, 2], fuse_21_6_data[:, 3], color='orange', label='ins+gnss_6dNoise')
        print('fuse_21_6 data:')
        evaluate(gt_data, fuse_21_6_data)

    if os.path.exists(fuse_21_18_data_path):
        fuse_21_18_data = np.loadtxt(fuse_21_18_data_path)
        ax.plot3D(fuse_21_18_data[:, 1], fuse_21_18_data[:, 2], fuse_21_18_data[:, 3], color='r', label='ins+gnss_18dNoise')
        print('fuse_21_18 data:')
        evaluate(gt_data, fuse_21_18_data)

    if os.path.exists(fuse_odo_data_path):
        fuse_odo_data = np.loadtxt(fuse_odo_data_path)
        ax.plot3D(fuse_odo_data[:, 1], fuse_odo_data[:, 2], fuse_odo_data[:, 3], color='purple', label='ins+gnss+vel')
        print('fuse_odo data:')
        evaluate(gt_data, fuse_odo_data)

    if os.path.exists(gps_data_path):
        gps_data = np.loadtxt(gps_data_path)
        ax.plot3D(gps_data[:data_size//200, 1], gps_data[:data_size//200, 2], gps_data[:data_size//200, 3], color='g', alpha=0.5, label='gps')
    if os.path.exists(ins_data_path):
        ins_data = np.loadtxt(ins_data_path)
        # ax.plot3D(ins_data[:, 1], ins_data[:, 2], ins_data[:, 3], color='y', label='ins')
    if os.path.exists(gins_data_path):
        gins_data = np.loadtxt(gins_data_path)
        ax.plot3D(gins_data[:, 1], gins_data[:, 2], gins_data[:, 3], color='c', label='gins')
    
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    plt.legend(loc='best')
    #ax.set_zlim(-100, 100)
    plt.show()

