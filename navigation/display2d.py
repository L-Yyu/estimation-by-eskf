import numpy as np
import matplotlib.pyplot as plt
import os

from tools import *

def load_txt_data(data_path):
    try:
        return np.loadtxt(data_path)
    except FileNotFoundError as err:
        print('this is a OSError: ' + str(err))

def evaluate(gt_data, fuse_data):
    rmse_x = np.sqrt(np.mean((fuse_data[:, 1] - gt_data[:, 1])**2))
    rmse_y = np.sqrt(np.mean((fuse_data[:, 2] - gt_data[:, 2])**2))
    rmse_z = np.sqrt(np.mean((fuse_data[:, 3] - gt_data[:, 3])**2))
    rmse_xy = np.sqrt(np.mean((fuse_data[:, 1:3] - gt_data[:, 1:3])**2))
    rmse_xyz = np.sqrt(np.mean((fuse_data[:, 1:4] - gt_data[:, 1:4])**2))
    i = np.argmax(np.abs(fuse_data[:, 1:3] - gt_data[:, 1:3]))
    print('position rmse x y z xy xyz: ', rmse_x, rmse_y, rmse_z,rmse_xy,rmse_xyz)
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
    data = 'sim_3'
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

    fig = plt.figure()
    ax = plt.axes()
    ax.set_title(data+' compare trajectory')

    if os.path.exists(gt_data_path):
        gt_data = np.loadtxt(gt_data_path)
        ax.plot(gt_data[:, 1], gt_data[:, 2], color='b', label='ground_truth')
        euler_gt = quaternion2euler(gt_data[:, 4:8])
    if os.path.exists(fuse_data_path):
        fuse_data = np.loadtxt(fuse_data_path)
        ax.plot(fuse_data[:, 1], fuse_data[:, 2], color='r', label='fuse')
        print('fuse data:')
        evaluate(gt_data, fuse_data)
    if os.path.exists(fuse_15_6_data_path):
        fuse_15_6_data = np.loadtxt(fuse_15_6_data_path)
        # ax.plot(fuse_15_6_data[:, 1], fuse_15_6_data[:, 2], color='purple', label='fuse_15_6')
        print('fuse_15_6 data:')
        #evaluate(gt_data, fuse_15_6_data)

    if os.path.exists(fuse_15_12_data_path):
        fuse_15_12_data = np.loadtxt(fuse_15_12_data_path)
        # ax.plot(fuse_15_12_data[:, 1], fuse_15_12_data[:, 2], color='c', label='fuse_15_12')
        print('fuse_15_12 data:')
        #evaluate(gt_data, fuse_15_12_data)

    if os.path.exists(fuse_21_6_data_path):
        fuse_21_6_data = np.loadtxt(fuse_21_6_data_path)
        #ax.plot(fuse_21_6_data[:, 1], fuse_21_6_data[:, 2], color='orange', label='fuse_6d_noise')
        print('fuse_21_6 data:')
        evaluate(gt_data, fuse_21_6_data)

    if os.path.exists(fuse_21_18_data_path):
        fuse_21_18_data = np.loadtxt(fuse_21_18_data_path)
        #ax.plot(fuse_21_18_data[:, 1], fuse_21_18_data[:, 2], color='r', label='fuse_18d_noise')
        print('fuse_21_18 data:')
        evaluate(gt_data, fuse_21_18_data)

    if os.path.exists(fuse_odo_data_path):
        fuse_odo_data = np.loadtxt(fuse_odo_data_path)
        #ax.plot(fuse_odo_data[:, 1], fuse_odo_data[:, 2], color='purple', label='ins+gnss+vel')
        print('fuse_odo data:')
        evaluate(gt_data, fuse_odo_data)

    if os.path.exists(gps_data_path):
        gps_data = np.loadtxt(gps_data_path)
        ax.plot(gps_data[:, 1], gps_data[:, 2], color='g', alpha=0.5, label='gps')
    if os.path.exists(ins_data_path):
        ins_data = np.loadtxt(ins_data_path)
        ax.plot(ins_data[:, 1], ins_data[:, 2], color='y', label='ins')
        print('ins data:')
        evaluate(gt_data, ins_data)
    # 设置横纵坐标的标签
    plt.xlabel('x(m)')
    plt.ylabel('y(m)')
    plt.legend(loc='best')
    plt.show()
    
    if False:
    # 绘制姿态角度误差    
        euler_gt = quaternion2euler(gt_data[:, 4:8])
        euler_fuse = quaternion2euler(fuse_data[:, 4:8])
        euler_ins = quaternion2euler(ins_data[:,4:8])
        d_roll = np.array([angle_diff(gt, fuse) for gt, fuse in zip(euler_gt[:, 0], euler_fuse[:, 0])])
        d_pitch = np.array([angle_diff(gt, fuse) for gt, fuse in zip(euler_gt[:, 1], euler_fuse[:, 1])])
        d_yaw = np.array([angle_diff(gt, fuse) for gt, fuse in zip(euler_gt[:, 2], euler_fuse[:, 2])])
        d_roll_ins = np.array([angle_diff(gt, ins) for gt, ins in zip(euler_gt[:, 0], euler_ins[:, 0])])
        d_pitch_ins = np.array([angle_diff(gt, ins) for gt, ins in zip(euler_gt[:, 1], euler_ins[:, 1])])
        d_yaw_ins = np.array([angle_diff(gt, ins) for gt, ins in zip(euler_gt[:, 2], euler_ins[:, 2])])
        
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        # 分别在三个子图上绘制数据
        axs[0].plot(d_roll, color='red', label='fuse')
        axs[0].plot(d_roll_ins, color='green', label='ins')
        axs[0].set_title("roll")
        axs[0].set_xlabel("time")
        axs[0].set_ylabel("d_roll(deg)")
        axs[0].grid(True)

        axs[1].plot(d_pitch, color='red', label='fuse')
        axs[1].plot(d_pitch_ins, color='green', label='ins')
        axs[1].set_title("pitch")
        axs[1].set_xlabel("time")
        axs[1].set_ylabel("d_pitch(deg)")
        axs[1].grid(True)

        axs[2].plot(d_yaw, color='red', label='fuse')
        axs[2].plot(d_yaw_ins, color='green', label='ins')
        axs[2].set_title("yaw")
        axs[2].set_xlabel("time")
        axs[2].set_ylabel("d_yaw(deg)")
        axs[2].grid(True)

        # 调整布局
        plt.tight_layout()

        plt.show()

    if False:
        
        fig, axs = plt.subplots(2, 1)
        # 分别在三个子图上绘制数据
        axs[0].plot(fuse_data[:,1]-gt_data[:,1], color='red', label='fuse')
        axs[0].plot(ins_data[:,1]-gt_data[:,1], color='green', label='ins')
        axs[0].set_xlabel("time(10ms)")
        axs[0].set_ylabel("d_north(m)")
        axs[0].grid(True)
        axs[0].legend(loc='best')

        axs[1].plot(fuse_data[:,2]-gt_data[:,2], color='red', label='fuse')
        axs[1].plot(ins_data[:,2]-gt_data[:,2], color='green', label='ins')
        axs[1].set_xlabel("time(10ms)")
        axs[1].set_ylabel("d_east(m)")
        axs[1].grid(True)
        axs[1].legend(loc='best')

        # 调整布局
        plt.tight_layout()

        plt.show()