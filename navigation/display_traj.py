import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits import mplot3d


def load_txt_data(data_path):
    try:
        return np.loadtxt(data_path)
    except FileNotFoundError as err:
        print('this is a OSError: ' + str(err))


if __name__ == "__main__":
    data_path = './data/test'
    fuse_data_path = data_path+'/fused.txt'
    gps_data_path = data_path+'/gps_measurement.txt'
    gt_data_path = data_path+'/gt.txt'
    ins_data_path = data_path+'/ins.txt'
    gins_data_path = data_path+'/fused_kfgins.txt'

    data_size = 300000
    gt_start_index = 10051
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_title('compare path')
    if os.path.exists(fuse_data_path):
        fuse_data = np.loadtxt(fuse_data_path)
        ax.plot3D(fuse_data[:data_size, 1], fuse_data[:data_size, 2], fuse_data[:data_size, 3], color='r', label='fuse_gps_imu')
    if os.path.exists(gps_data_path):
        gps_data = np.loadtxt(gps_data_path)
        ax.plot3D(gps_data[:data_size//200, 1], gps_data[:data_size//200, 2], gps_data[:data_size//200, 3], color='g', alpha=0.5, label='gps')
    if os.path.exists(gt_data_path):
        gt_data = np.loadtxt(gt_data_path)
        ax.plot3D(gt_data[gt_start_index:gt_start_index+data_size, 1], gt_data[gt_start_index:gt_start_index+data_size, 2], gt_data[gt_start_index:gt_start_index+data_size, 3], color='b', label='ground_truth')
    if os.path.exists(ins_data_path):
        ins_data = np.loadtxt(ins_data_path)
        # ax.plot3D(ins_data[:, 1], ins_data[:, 2], ins_data[:, 3], color='y', label='ins')
    if os.path.exists(gins_data_path):
        gins_data = np.loadtxt(gins_data_path)
        ax.plot3D(gins_data[:data_size, 1], gins_data[:data_size, 2], gins_data[:data_size, 3], color='c', label='gins')
    plt.legend(loc='best')
    #ax.set_zlim(-100, 100)
    plt.show()
