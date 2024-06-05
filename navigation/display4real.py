import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits import mplot3d
from tools import *
from display2d import evaluate
def load_txt_data(data_path):
    try:
        return np.loadtxt(data_path)
    except FileNotFoundError as err:
        print('this is a OSError: ' + str(err))

if __name__ == "__main__":
    data = 'i300'
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

    data_size = 105400 #40000 #test200000
    gt_start_index = 0 #5425#test10052
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_title('fuse results comparison')
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
        #ax.plot3D(fuse_data[:, 1], fuse_data[:, 2], fuse_data[:, 3], color='r', label='fuse')
        # 检查数据是否对齐
        print(fuse_data[0,0],gt_data[0,0])
        print(fuse_data[-1,0],gt_data[-1,0])
        
        print('fuse data:')
        evaluate(gt_data, fuse_data)

    if os.path.exists('./data/i300/fused_18_1.txt'):
        data1 = np.loadtxt('./data/i300/fused_18_1.txt')
        ax.plot3D(data1[:, 1], data1[:, 2], data1[:, 3], color='r', label='fuse_18d_noise',alpha=0.5)
        evaluate(gt_data, data1)
    if os.path.exists('./data/i300/fused_18_0.txt'):
        data2 = np.loadtxt('./data/i300/fused_18_0.txt')
        ax.plot3D(data2[:, 1], data2[:, 2], data2[:, 3], color='orange', label='fuse_6d_noise', alpha=0.5)
        evaluate(gt_data, data2)

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
        #ax.plot3D(gps_data[:data_size//200, 1], gps_data[:data_size//200, 2], gps_data[:data_size//200, 3], color='g', alpha=0.5, label='gps')
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

    if True:
        #位置误差可视化
        fig, axs = plt.subplots(3, 1)
        # 分别在三个子图上绘制数据
        axs[0].plot(data1[:-40000,1]-gt_data[:-40000,1], color='red', label='fuse_compensate')
        axs[0].plot(data2[:-40000,1]-gt_data[:-40000,1], color='green', label='fuse_no_compensate')
        axs[0].set_xlabel("time(5ms)")
        axs[0].set_ylabel("d_north(m)")
        axs[0].grid(True)
        axs[0].legend(loc='best')

        axs[1].plot(data1[:-40000,2]-gt_data[:-40000,2], color='red', label='fuse_compensate')
        axs[1].plot(data2[:-40000,2]-gt_data[:-40000,2], color='green', label='fuse_no_compensate')
        axs[1].set_xlabel("time(5ms)")
        axs[1].set_ylabel("d_east(m)")
        axs[1].grid(True)
        axs[1].legend(loc='best')
        
        axs[2].plot(data1[:-40000,3]-gt_data[:-40000,3], color='red', label='fuse_compensate')
        axs[2].plot(data2[:-40000,3]-gt_data[:-40000,3], color='green', label='fuse_no_compensate')
        axs[2].set_xlabel("time(5ms)")
        axs[2].set_ylabel("d_down(m)")
        axs[2].grid(True)
        axs[2].legend(loc='best')

        # 调整布局
        plt.tight_layout()

        plt.show()

        
    if True:
        #姿态角误差可视化
        euler_gt = quaternion2euler(gt_data[:, 4:8])
        euler_fuse1 = quaternion2euler(data1[:, 4:8])
        d_roll1 = np.array([angle_diff(gt, fuse) for gt, fuse in zip(euler_gt[:, 0], euler_fuse1[:, 0])])
        d_pitch1 = np.array([angle_diff(gt, fuse) for gt, fuse in zip(euler_gt[:, 1], euler_fuse1[:, 1])])
        d_yaw1 = np.array([angle_diff(gt, fuse) for gt, fuse in zip(euler_gt[:, 2], euler_fuse1[:, 2])])

        euler_fuse2 = quaternion2euler(data2[:, 4:8])
        d_roll2 = np.array([angle_diff(gt, fuse) for gt, fuse in zip(euler_gt[:, 0], euler_fuse2[:, 0])])
        d_pitch2 = np.array([angle_diff(gt, fuse) for gt, fuse in zip(euler_gt[:, 1], euler_fuse2[:, 1])])
        d_yaw2 = np.array([angle_diff(gt, fuse) for gt, fuse in zip(euler_gt[:, 2], euler_fuse2[:, 2])])
        
        fig, axs = plt.subplots(3, 1)
        # 分别在三个子图上绘制数据
        axs[0].plot(d_roll1[:-40000], color='red', label='fuse_compensate')
        axs[0].plot(d_roll2[:-40000], color='green', label='fuse_no_compensate')
        axs[0].set_xlabel("time(5ms)")
        axs[0].set_ylabel("d_roll(deg)")
        axs[0].grid(True)
        axs[0].legend(loc='best')

        axs[1].plot(d_pitch1[:-40000], color='red', label='fuse_compensate')
        axs[1].plot(d_pitch2[:-40000], color='green', label='fuse_no_compensate')
        axs[1].set_xlabel("time(5ms)")
        axs[1].set_ylabel("d_pitch(deg)")
        axs[1].grid(True)
        axs[1].legend(loc='best')
        
        axs[2].plot(d_yaw1[:-40000], color='red', label='fuse_compensate')
        axs[2].plot(d_yaw2[:-40000], color='green', label='fuse_no_compensate')
        axs[2].set_xlabel("time(5ms)")
        axs[2].set_ylabel("d_yaw(deg)")
        axs[2].grid(True)
        axs[2].legend(loc='best')

        # 调整布局
        plt.tight_layout()

        plt.show()