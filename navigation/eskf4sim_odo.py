import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import os
import yaml
import time
from tqdm import tqdm
from data import IMUData, GNSSData
from tools import BuildSkewSymmetricMatrix, rv2rm, rm2quaternion, euler2rm, euler2quaternion, rv2rm
from earth import lla2ned, ned2lla, GetGravity
from observability import ObservabilityAnalysis


ID_STATE_P = 0
ID_STATE_V = 3
ID_STATE_PHI = 6
ID_STATE_BG = 9
ID_STATE_BA = 12
ID_STATE_SG = 15
ID_STATE_SA = 18

ID_MEASUREMENT_P = 0

D2R = np.pi/180

class ESKF(object):

    def __init__(self, config, states_rank, noise_rank) -> None:
        self.dim_state = states_rank         #   15:p v phi bg ba    21:p v phi bg ba sg sa
        self.dim_state_noise = noise_rank   #   6:w_phi w_v     12: w_phi w_v bg ba     18: w_phi w_v bg ba sg sa
        self.dim_measurement = 3
        self.dim_measurement_noise = 3
        # dX = FX + BW(Q)  Y = GX + CN(R)   Cov(X) = P
        self.X = np.zeros((self.dim_state, 1))
        self.Y = np.zeros((self.dim_measurement, 1))
        self.F = np.zeros((self.dim_state, self.dim_state))
        self.B = np.zeros((self.dim_state, self.dim_state_noise))
        self.Q = np.zeros((self.dim_state_noise, self.dim_state_noise))
        self.P = np.zeros((self.dim_state, self.dim_state))
        self.K = np.zeros((self.dim_state, self.dim_measurement))
        self.C = np.identity(self.dim_measurement_noise)
        self.G = np.zeros((self.dim_measurement, self.dim_state))
        self.R = np.zeros((self.dim_measurement, self.dim_measurement))
        self.Ft = np.zeros((self.dim_state, self.dim_state))
        
        # 初始化状态方差
        self.SetP(config['init_position_std'], config['init_velocity_std'], config['init_rotation_std'],
                  config['init_bg_std'], config['init_ba_std'], config['init_sg_std'], config['init_sa_std'])
        # 初始化imu过程噪声方差
        self.SetQ(config['arw'], config['vrw'], config['bg_std'], config['ba_std'], config['sg_std'], config['sa_std'], config['corr_time'])
        # 初始化gnss观测噪声方差
        self.SetR(config['gps_position_x_std'], config['gps_position_y_std'], config['gps_position_z_std'])

        # 初始化部分参数
        self.corr_time_ = config['corr_time']
        self.ref_pos_lla_ = np.array(config['init_position_lla']).reshape(3,1)
        self.g_n = np.array([0, 0, GetGravity(self.ref_pos_lla_)]).reshape((3,1))
        
        # 初始化状态
        self.InitState(config)
        

    def SetQ(self, arw, vrw, bg_std, ba_std, sg_std, sa_std, corr_time):
        # 设置imu误差
        # 转换为标准单位
        arw = np.array(arw) * D2R / 60.0   # deg/sqrt(h) -> rad/sqrt(s)
        vrw = np.array(vrw) / 60.0  # m/s/sqrt(h) -> m/s/sqrt(s)
        bg_std = np.array(bg_std) * D2R / 3600.0  # deg/h -> rad/s
        ba_std = np.array(ba_std) * 1e-5    # mGal -> m/s^2
        sg_std = np.array(sg_std) * 1e-6    # ppm -> 1
        sa_std = np.array(sa_std) * 1e-6    # ppm -> 1
        corr_time = np.array(corr_time) * 3600.0  # h -> s
        # self.Q = np.zeros((self.dim_state_noise, self.dim_state_noise))
        if self.dim_state == 15 and self.dim_state_noise == 6:
            self.Q[0:3, 0:3] = np.eye(3) * arw * arw
            self.Q[3:6, 3:6] = np.eye(3) * vrw * vrw
            # self.Q[0:3, 0:3] = np.eye(3) * 1e-2 * 1e-2
            # self.Q[3:6, 3:6] = np.eye(3) * 1e-1 * 1e-1
        elif self.dim_state==15 and self.dim_state_noise == 12:
            self.Q[0:3, 0:3] = np.eye(3) * arw * arw
            self.Q[3:6, 3:6] = np.eye(3) * vrw * vrw
            self.Q[6:9, 6:9] = np.eye(3) * bg_std * bg_std* 2 / corr_time
            self.Q[9:12, 9:12] = np.eye(3) * ba_std * ba_std* 2 / corr_time
        elif self.dim_state==21 and self.dim_state_noise == 6:
            self.Q[0:3, 0:3] = np.eye(3) * arw * arw
            self.Q[3:6, 3:6] = np.eye(3) * vrw * vrw
        elif self.dim_state==21 and self.dim_state_noise == 18:
            self.Q[0:3, 0:3] = np.eye(3) * arw * arw
            self.Q[3:6, 3:6] = np.eye(3) * vrw * vrw
            self.Q[6:9, 6:9] = np.eye(3) * bg_std * bg_std* 2 / corr_time
            self.Q[9:12, 9:12] = np.eye(3) * ba_std * ba_std* 2 / corr_time
            self.Q[12:15, 12:15] = np.eye(3) * sg_std * sg_std* 2 / corr_time
            self.Q[15:18, 15:18] = np.eye(3) * sa_std * sa_std* 2 / corr_time
        else:
            print('SetQ failed')
            return

    def SetR(self, position_x_std, position_y_std, position_z_std):
        # self.R = np.zeros((self.dim_measurement, self.dim_measurement))
        self.R[0, 0] =  position_x_std * position_x_std
        self.R[1, 1] =  position_y_std * position_y_std
        self.R[2, 2] =  position_z_std * position_z_std

    def SetP(self, init_posi_std, init_vel_std, init_ori_std, init_bg_std, init_ba_std, init_sg_std, init_sa_std):
        # 设置初始状态协方差矩阵
        init_posi_std = np.array(init_posi_std)  # m
        init_vel_std = np.array(init_vel_std)   # m/s
        init_ori_std = np.array(init_ori_std) * D2R   # deg -> rad
        init_bg_std = np.array(init_bg_std) * D2R / 3600.0    # deg/h -> rad/s
        init_ba_std = np.array(init_ba_std) * 1e-5  # mGal -> m/s^2
        init_sg_std = np.array(init_sg_std) * 1e-6  # ppm -> 1
        init_sa_std = np.array(init_sa_std) * 1e-6  # ppm -> 1
        # self.P = np.zeros((self.dim_state, self.dim_state))
        if self.dim_state == 15:
            self.P[ID_STATE_P:ID_STATE_P+3, ID_STATE_P:ID_STATE_P+3] = np.eye(3) * init_posi_std * init_posi_std
            self.P[ID_STATE_V:ID_STATE_V+3, ID_STATE_V:ID_STATE_V+3] = np.eye(3) * init_vel_std * init_vel_std
            self.P[ID_STATE_PHI:ID_STATE_PHI+3, ID_STATE_PHI:ID_STATE_PHI+3] = np.eye(3) * init_ori_std * init_ori_std
            self.P[ID_STATE_BG:ID_STATE_BG+3, ID_STATE_BG:ID_STATE_BG+3] = np.eye(3) * init_bg_std * init_bg_std
            self.P[ID_STATE_BA:ID_STATE_BA+3, ID_STATE_BA:ID_STATE_BA+3] = np.eye(3) * init_ba_std * init_ba_std    
        elif self.dim_state == 21:
            self.P[ID_STATE_P:ID_STATE_P+3, ID_STATE_P:ID_STATE_P+3] = np.eye(3) * init_posi_std * init_posi_std
            self.P[ID_STATE_V:ID_STATE_V+3, ID_STATE_V:ID_STATE_V+3] = np.eye(3) * init_vel_std * init_vel_std
            self.P[ID_STATE_PHI:ID_STATE_PHI+3, ID_STATE_PHI:ID_STATE_PHI+3] = np.eye(3) * init_ori_std * init_ori_std
            self.P[ID_STATE_BG:ID_STATE_BG+3, ID_STATE_BG:ID_STATE_BG+3] = np.eye(3) * init_bg_std * init_bg_std
            self.P[ID_STATE_BA:ID_STATE_BA+3, ID_STATE_BA:ID_STATE_BA+3] = np.eye(3) * init_ba_std * init_ba_std  
            self.P[ID_STATE_SG:ID_STATE_SG+3, ID_STATE_SG:ID_STATE_SG+3] = np.eye(3) * init_sg_std * init_sg_std
            self.P[ID_STATE_SA:ID_STATE_SA+3, ID_STATE_SA:ID_STATE_SA+3] = np.eye(3) * init_sa_std * init_sa_std
        else:
            print('SetP failed')
            return
    def SetG(self):
        # 设置观测矩阵
        # self.G = np.zeros((self.dim_measurement, self.dim_state))
        self.G[ID_MEASUREMENT_P:ID_MEASUREMENT_P+3, ID_MEASUREMENT_P:ID_MEASUREMENT_P+3] = np.eye(3)

    def InitState(self, config):
        self.pos_ = np.array([0, 0, 0]).reshape(3,1)
        self.pos_lla_ = np.array(config['init_position_lla']).reshape(3,1)

        self.velocity_ = np.array(config['init_velocity']).reshape(3,1)
        
        euler = np.array(config['init_attitude'])
        self.quat_ = euler2quaternion(euler)
        self.rotation_matrix_ = euler2rm(euler)

        self.gyro_bias_ = np.array(config['init_gyro_bias']).reshape(3,1)
        self.accel_bias_ = np.array(config['init_accel_bias']).reshape(3,1)

        if self.dim_state == 21:
            self.gyro_scale_ = np.array(config['init_gyro_scale']).reshape(3,1)
            self.accel_scale_ = np.array(config['init_accel_scale']).reshape(3,1)

    def Predict(self, last_imu_data, curr_imu_data):
        delta_t = curr_imu_data.imu_time - last_imu_data.imu_time
        # 根据上一时刻状态，计算状态转移矩阵F_k-1, B_k-1
        self.UpdateErrorState(delta_t, last_imu_data)

        # 姿态更新
         # gyro数据补偿
        unbias_gyro_0 = last_imu_data.imu_angle_vel - self.gyro_bias_   #[deg/s]
        unbias_gyro_1 = curr_imu_data.imu_angle_vel - self.gyro_bias_
        if self.dim_state == 21:
            unbias_gyro_0 = (np.ones((3,1)) + self.gyro_scale_)*unbias_gyro_0
            unbias_gyro_1 = (np.ones((3,1)) + self.gyro_scale_)*unbias_gyro_1
        delta_theta = 0.5 * (unbias_gyro_0 + unbias_gyro_1) * delta_t
        rotation_vector = delta_theta
        # 基于旋转矩阵的更新算法
        last_rotation_matrix = self.rotation_matrix_
        delta_rotation_matrix = rv2rm(rotation_vector)
        curr_rotation_matrix = last_rotation_matrix @ delta_rotation_matrix
        self.rotation_matrix_ = curr_rotation_matrix
        self.quat_ = rm2quaternion(curr_rotation_matrix)

        # 速度更新
        unbias_accel_n_0 = last_rotation_matrix @ (last_imu_data.imu_linear_acceleration - self.accel_bias_) + self.g_n
        unbias_accel_n_1 = curr_rotation_matrix @ (curr_imu_data.imu_linear_acceleration - self.accel_bias_) + self.g_n
        if self.dim_state == 21:
            pass
        last_vel_n = self.velocity_
        curr_vel_n = last_vel_n + delta_t * 0.5 * (unbias_accel_n_0 + unbias_accel_n_1)
        self.velocity_ = curr_vel_n
        # print('vb',self.rotation_matrix_.T@self.velocity_)
        # print('vb_measure',euler2rm(curr_imu_data.true_euler).T @ curr_imu_data.true_vel_n)
        # 位置更新
        self.pos_ = self.pos_ + 0.5 * delta_t * (curr_vel_n + last_vel_n) + 0.25 * (unbias_accel_n_0 + unbias_accel_n_1) * delta_t * delta_t
        self.pos_lla_ = ned2lla(self.pos_, self.ref_pos_lla_)
        # self.g_ = np.array([0, 0, -GetGravity(self.pos_lla_)]).reshape((3,1))
        
        if False:
            print('imu_time: ', curr_imu_data.imu_time)
            print('pos: ', self.pos_)
            print('vel: ', self.velocity_)
            print('rm: ', self.quat_)
        
        # self.CorrectOdom(curr_imu_data)
        # self.CorrectVb1(curr_imu_data)


    def UpdateErrorState(self, delta_t, imu_data):
        accel_b = imu_data.imu_linear_acceleration 
        accel_n = self.rotation_matrix_ @ accel_b
        w_ib_b = imu_data.imu_angle_vel
        w_ib_n = self.rotation_matrix_ @ w_ib_b
        
        # 状态转移矩阵
        F_23 = BuildSkewSymmetricMatrix(accel_n)
        F_33 = -BuildSkewSymmetricMatrix(np.array([0,0,0]).reshape(3,1)) # w_in_n or w_ie_n

        if self.dim_state == 15 and self.dim_state_noise == 6:
            self.F[ID_STATE_P:ID_STATE_P+3, ID_STATE_V:ID_STATE_V+3] = np.eye(3)
            self.F[ID_STATE_V:ID_STATE_V+3, ID_STATE_PHI:ID_STATE_PHI+3] = F_23
            self.F[ID_STATE_PHI:ID_STATE_PHI+3, ID_STATE_PHI:ID_STATE_PHI+3] = F_33
            self.F[ID_STATE_V:ID_STATE_V+3, ID_STATE_BA:ID_STATE_BA+3] = self.rotation_matrix_
            self.F[ID_STATE_PHI:ID_STATE_PHI+3, ID_STATE_BG:ID_STATE_BG+3] = -self.rotation_matrix_
        elif self.dim_state == 15 and self.dim_state_noise == 12:
            self.F[ID_STATE_P:ID_STATE_P+3, ID_STATE_V:ID_STATE_V+3] = np.eye(3)
            self.F[ID_STATE_V:ID_STATE_V+3, ID_STATE_PHI:ID_STATE_PHI+3] = F_23
            self.F[ID_STATE_PHI:ID_STATE_PHI+3, ID_STATE_PHI:ID_STATE_PHI+3] = F_33
            self.F[ID_STATE_V:ID_STATE_V+3, ID_STATE_BA:ID_STATE_BA+3] = self.rotation_matrix_
            self.F[ID_STATE_PHI:ID_STATE_PHI+3, ID_STATE_BG:ID_STATE_BG+3] = -self.rotation_matrix_
            self.F[ID_STATE_BG:ID_STATE_BG+3, ID_STATE_BG:ID_STATE_BG+3] = (-1/self.corr_time_)*np.eye(3)
            self.F[ID_STATE_BA:ID_STATE_BA+3, ID_STATE_BA:ID_STATE_BA+3] = (-1/self.corr_time_)*np.eye(3)
        elif self.dim_state == 21 and self.dim_state_noise == 6:
            self.F[ID_STATE_P:ID_STATE_P+3, ID_STATE_V:ID_STATE_V+3] = np.eye(3)
            self.F[ID_STATE_V:ID_STATE_V+3, ID_STATE_PHI:ID_STATE_PHI+3] = F_23
            self.F[ID_STATE_PHI:ID_STATE_PHI+3, ID_STATE_PHI:ID_STATE_PHI+3] = F_33
            self.F[ID_STATE_V:ID_STATE_V+3, ID_STATE_BA:ID_STATE_BA+3] = self.rotation_matrix_
            self.F[ID_STATE_PHI:ID_STATE_PHI+3, ID_STATE_BG:ID_STATE_BG+3] = -self.rotation_matrix_
            self.F[ID_STATE_V:ID_STATE_V+3, ID_STATE_SA:ID_STATE_SA+3] = self.rotation_matrix_ @ np.diag(accel_b.reshape(3))
            self.F[ID_STATE_PHI:ID_STATE_PHI+3, ID_STATE_SG:ID_STATE_SG+3] = -self.rotation_matrix_ @ np.diag(w_ib_b.reshape(3))
        elif self.dim_state == 21 and self.dim_state_noise == 18:
            self.F[ID_STATE_P:ID_STATE_P+3, ID_STATE_V:ID_STATE_V+3] = np.eye(3)
            self.F[ID_STATE_V:ID_STATE_V+3, ID_STATE_PHI:ID_STATE_PHI+3] = F_23
            self.F[ID_STATE_PHI:ID_STATE_PHI+3, ID_STATE_PHI:ID_STATE_PHI+3] = F_33
            self.F[ID_STATE_V:ID_STATE_V+3, ID_STATE_BA:ID_STATE_BA+3] = self.rotation_matrix_
            self.F[ID_STATE_PHI:ID_STATE_PHI+3, ID_STATE_BG:ID_STATE_BG+3] = -self.rotation_matrix_
            self.F[ID_STATE_V:ID_STATE_V+3, ID_STATE_SA:ID_STATE_SA+3] = self.rotation_matrix_ @ np.diag(accel_b.reshape(3))
            self.F[ID_STATE_PHI:ID_STATE_PHI+3, ID_STATE_SG:ID_STATE_SG+3] = -self.rotation_matrix_ @ np.diag(w_ib_b.reshape(3))
            self.F[ID_STATE_BG:ID_STATE_BG+3, ID_STATE_BG:ID_STATE_BG+3] = (-1/self.corr_time_)*np.eye(3)
            self.F[ID_STATE_BA:ID_STATE_BA+3, ID_STATE_BA:ID_STATE_BA+3] = (-1/self.corr_time_)*np.eye(3)
            self.F[ID_STATE_SG:ID_STATE_SG+3, ID_STATE_SG:ID_STATE_SG+3] = (-1/self.corr_time_)*np.eye(3)
            self.F[ID_STATE_SA:ID_STATE_SA+3, ID_STATE_SA:ID_STATE_SA+3] = (-1/self.corr_time_)*np.eye(3)
        else:
            print('wrong dim_state or dim_state_noise')

        # 噪声驱动矩阵
        if self.dim_state_noise == 6:
            self.B[ID_STATE_V:ID_STATE_V+3, 3:6] = self.rotation_matrix_
            self.B[ID_STATE_PHI:ID_STATE_PHI+3, 0:3] = -self.rotation_matrix_
        elif self.dim_state_noise == 12:
            self.B[ID_STATE_V:ID_STATE_V+3, 3:6] = self.rotation_matrix_
            self.B[ID_STATE_PHI:ID_STATE_PHI+3, 0:3] = -self.rotation_matrix_
            self.B[ID_STATE_BG:ID_STATE_BG+3, 6:9] = np.eye(3)
            self.B[ID_STATE_BA:ID_STATE_BA+3, 9:12] = np.eye(3)
        elif self.dim_state_noise == 18:
            self.B[ID_STATE_V:ID_STATE_V+3, 3:6] = self.rotation_matrix_
            self.B[ID_STATE_PHI:ID_STATE_PHI+3, 0:3] = -self.rotation_matrix_
            self.B[ID_STATE_BG:ID_STATE_BG+3, 6:9] = np.eye(3)
            self.B[ID_STATE_BA:ID_STATE_BA+3, 9:12] = np.eye(3)
            self.B[ID_STATE_SG:ID_STATE_SG+3, 12:15] = np.eye(3)
            self.B[ID_STATE_SA:ID_STATE_SA+3, 15:18] = np.eye(3)
        
        # 离散化1 (已有数据在此方式下计算)
        #Phi = np.eye(self.dim_state) + self.F * delta_t
        #Bk = self.B * delta_t
        #Qd = Bk @ self.Q @ Bk.T
        # 用于可观测性分析
        # self.Fo = self.F * delta_t
        
        # 离散化2
        Phi = np.eye(self.dim_state) + self.F * delta_t
        Qd = self.B @ self.Q @ self.B.T
        Qd = (Phi @ Qd @ Phi.T + Qd) * delta_t / 2
        # 用于可观测性分析
        self.Fo = Phi

        # KF prediction
        self.X = Phi @ self.X
        self.P = Phi @ self.P @ Phi.T + Qd


    def Correct(self, curr_gnss_data):
        self.G[ID_MEASUREMENT_P:ID_MEASUREMENT_P+3, ID_STATE_P:ID_STATE_P+3] = np.eye(3)

        self.Y = self.pos_ - lla2ned(curr_gnss_data.position_lla, self.ref_pos_lla_)
        self.K = self.P @ self.G.T @ np.linalg.inv(self.G @ self.P @ self.G.T + self.C @ self.R @ self.C.T)
        self.P = (np.eye(self.dim_state) - self.K @ self.G) @ self.P
        self.X = self.X + self.K @ (self.Y - self.G @ self.X)
        
        # 误差状态反馈至完整状态
        self.StateFeedback()
        self.pos_lla_ = ned2lla(self.pos_, self.ref_pos_lla_)

        # 误差状态置零
        self.ResetState()
        # self.CorrectVn(curr_gnss_data)
    

    def CorrectVn(self, curr_gnss_data):
        # 使用gps的n系速度观测数据进行状态更新
        self.G1 = np.zeros((3, self.dim_state))
        self.G1[0:3, ID_STATE_V:ID_STATE_V+3] = np.eye(3)
        self.C1 = np.identity(3)
        self.R1 = np.zeros((3,3))
        self.R1[0,0] = 0.1 * 0.1
        self.R1[1,1] = 0.01
        self.R1[2,2] = 0.01

        self.Y1 = self.velocity_ - curr_gnss_data.true_velocity 
        self.K1 = self.P @ self.G1.T @ np.linalg.inv(self.G1 @ self.P @ self.G1.T + self.C1 @ self.R1 @ self.C1.T)
        self.P = (np.eye(self.dim_state) - self.K1 @ self.G1) @ self.P
        self.X = self.X + self.K1 @ (self.Y1 - self.G1 @ self.X)
        
        self.StateFeedback()
        self.pos_lla_ = ned2lla(self.pos_, self.ref_pos_lla_)

        # 误差状态置零
        self.ResetState()

    def CorrectVb(self, curr_imu_data):
        # 使用imu的b系速度观测数据进行状态更新
        self.G1 = np.zeros((3, self.dim_state))
        self.G1[0:3, ID_STATE_V:ID_STATE_V+3] = self.rotation_matrix_.T
        self.G1[0:3, ID_STATE_PHI:ID_STATE_PHI+3] = -self.rotation_matrix_.T @ BuildSkewSymmetricMatrix(self.velocity_)
        self.C1 = np.identity(3)
        self.R1 = np.zeros((3,3))
        self.R1[0,0] = 0.01
        self.R1[1,1] = 0.01
        self.R1[2,2] = 0.01

        self.Y1 = self.rotation_matrix_.T @ self.velocity_ - np.array([curr_imu_data.odom_vel, 0, 0]).reshape(3,1)
        self.K1 = self.P @ self.G1.T @ np.linalg.inv(self.G1 @ self.P @ self.G1.T + self.C1 @ self.R1 @ self.C1.T)
        self.P = (np.eye(self.dim_state) - self.K1 @ self.G1) @ self.P
        self.X = self.X + self.K1 @ (self.Y1 - self.G1 @ self.X)
        
        self.StateFeedback()
        self.pos_lla_ = ned2lla(self.pos_, self.ref_pos_lla_)

        # 误差状态置零
        self.ResetState()

    def CorrectVbGnss(self, curr_imu_data, curr_gnss_data):
        # 同时用位置和速度进行状态更新
        self.G1 = np.zeros((6, self.dim_state))
        self.G1[0:3, 0:3] = np.eye(3)
        self.G1[3:6, ID_STATE_V:ID_STATE_V+3] = self.rotation_matrix_.T
        self.G1[3:6, ID_STATE_PHI:ID_STATE_PHI+3] = -self.rotation_matrix_.T @ BuildSkewSymmetricMatrix(self.velocity_)
        self.C1 = np.identity(6)
        self.R1 = np.zeros((6,6))
        self.R1[0,0] = 25
        self.R1[1,1] = 25
        self.R1[2,2] = 49
        self.R1[3,3] = 0.01
        self.R1[4,4] = 0.01
        self.R1[5,5] = 0.01
        self.Y1 = np.zeros((6,1))

        self.Y1[0:3, :] = self.pos_ - lla2ned(curr_gnss_data.position_lla, self.ref_pos_lla_)
        #self.Y1[3:6, :] = self.rotation_matrix_.T @ self.velocity_ - euler2rm(curr_imu_data.true_euler).T @ curr_imu_data.true_vel_n
        self.Y1[3:6, :] = self.rotation_matrix_.T @ self.velocity_ - np.array([curr_imu_data.odom_vel, 0, 0]).reshape(3,1)
        self.K1 = self.P @ self.G1.T @ np.linalg.inv(self.G1 @ self.P @ self.G1.T + self.C1 @ self.R1 @ self.C1.T)
        self.P = (np.eye(self.dim_state) - self.K1 @ self.G1) @ self.P
        self.X = self.X + self.K1 @ (self.Y1 - self.G1 @ self.X)
        
        self.StateFeedback()
        self.pos_lla_ = ned2lla(self.pos_, self.ref_pos_lla_)

        # 误差状态置零
        self.ResetState()

    def StateFeedback(self):
        self.pos_ = self.pos_ - self.X[ID_STATE_P:ID_STATE_P+3, :]
        self.velocity_ = self.velocity_ - self.X[ID_STATE_V:ID_STATE_V+3, :]
        cp_n = rv2rm(self.X[ID_STATE_PHI:ID_STATE_PHI+3, :])
        self.rotation_matrix_ = cp_n @ self.rotation_matrix_
        self.quat_ = rm2quaternion(self.rotation_matrix_)
        self.gyro_bias_ = self.gyro_bias_ + self.X[ID_STATE_BG:ID_STATE_BG+3, :]
        self.accel_bias_ = self.accel_bias_ + self.X[ID_STATE_BA:ID_STATE_BA+3, :]
        if self.dim_state == 21:
            self.gyro_scale_ = self.gyro_scale_ + self.X[ID_STATE_SG:ID_STATE_SG+3, :]
            self.accel_scale_ = self.accel_scale_ + self.X[ID_STATE_SA:ID_STATE_SA+3, :]
        # self.g_ = np.array([0, 0, -GetGravity(self.pos_lla_)]).reshape((3,1))
    def ResetState(self):
        self.X = np.zeros((self.dim_state, 1))

    def SaveData(self, file, imu_data):
        file.write(str(imu_data.imu_time)+' ')
        for i in range(3):
            file.write(str(eskf.pos_[i][0])+' ')
        for i in range(4):
            if i < 3:
                file.write(str(eskf.quat_[i])+' ')
            else:
                file.write(str(eskf.quat_[i])+'\n')

    def SaveGnssData(self, file, gnss_data):
        curr_position_ned = lla2ned(gnss_data.position_lla ,self.ref_pos_lla_)
        file.write(str(gnss_data.gnss_time)+' ')
        for i in range(3):
            file.write(str(curr_position_ned[i][0])+' ')
        file.write('0 0 0 1\n')

    def SaveGTData(self, file, imu_data):
        gt_pos_ned = lla2ned(imu_data.gt_pos_lla, self.ref_pos_lla_)
        file.write(str(imu_data.imu_time)+' ')
        for i in range(3):
            file.write(str(gt_pos_ned[i][0])+' ')
        for i in range(4):
            if i < 3:
                file.write(str(imu_data.gt_quat[i])+' ')
            else:
                file.write(str(imu_data.gt_quat[i])+'\n')
        
if __name__ == "__main__":
    data_path = './data/sim_1'
    # 对于sim 21 6 = 15 6  21 18 = 15 12 
    states_rank = 21
    noise_rank = 6
    is_obs_analysis = True
    if is_obs_analysis:
        OA = ObservabilityAnalysis(states_rank)
    # load configuration
    config_path = os.path.join(data_path,'config.yaml')
    with open(config_path,encoding='utf-8') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
        print(config)

    # open file
    fuse_file_name = os.path.join(data_path,'fused.txt')
    gps_file_name = os.path.join(data_path,'gps_measurement.txt')
    gt_file_name = os.path.join(data_path,'gt.txt')
    # ins_file_name = os.path.join(data_path,'ins.txt')
    
    fuse_file = open(fuse_file_name,'w')
    gps_file = open(gps_file_name,'w')
    gt_file = open(gt_file_name,'w')
    # ins_file = open(ins_file_name,'w')
    
    # imu gps fuse
    tick_start = time.time()
    eskf = ESKF(config, states_rank, noise_rank)

    imu_data_queue = IMUData.read_imu_data(data_path)
    gnss_data_queue = GNSSData.read_gnss_data(data_path)
    curr_imu_data = imu_data_queue.get()
    curr_gnss_data = gnss_data_queue.get()

    start_time = curr_imu_data.imu_time
    data_size = imu_data_queue.qsize()
    for i in tqdm(range(data_size), desc="fusing IMU and GNSS"):
        if imu_data_queue.empty() or gnss_data_queue.empty():
            break
        last_imu_data = curr_imu_data
        curr_imu_data = imu_data_queue.get()
        if abs(curr_gnss_data.gnss_time - last_imu_data.imu_time) < 0.001:
            # 靠近上一时刻
            eskf.CorrectVb(curr_imu_data)
            eskf.Correct(curr_gnss_data)
            #eskf.Correct(curr_imu_data, curr_gnss_data)
            eskf.SaveGnssData(gps_file, curr_gnss_data)
            curr_gnss_data = gnss_data_queue.get()

            eskf.Predict(last_imu_data, curr_imu_data)
        elif abs(curr_gnss_data.gnss_time - curr_imu_data.imu_time) < 0.001:
            # 靠近当前时刻
            eskf.Predict(last_imu_data, curr_imu_data)

            eskf.CorrectVb(curr_imu_data)
            eskf.Correct(curr_gnss_data)
            #eskf.CorrectVb2(curr_imu_data, curr_gnss_data)
            eskf.SaveGnssData(gps_file, curr_gnss_data)
            curr_gnss_data = gnss_data_queue.get()
        elif (curr_gnss_data.gnss_time > last_imu_data.imu_time) and (curr_gnss_data.gnss_time < curr_imu_data.imu_time):
            # 内插到gnss时刻
            print('interpolate')
            mid_imu_data = IMUData.Interpolate(last_imu_data, curr_imu_data, curr_gnss_data.gnss_time)
            eskf.Predict(last_imu_data, mid_imu_data)
            
            eskf.CorrectVb(mid_imu_data)
            eskf.Correct(curr_gnss_data)
            #eskf.CorrectVb2(mid_imu_data, curr_gnss_data)
            eskf.SaveGnssData(gps_file, curr_gnss_data)
            curr_gnss_data = gnss_data_queue.get()

            eskf.Predict(mid_imu_data, curr_imu_data)
        else:
            eskf.Predict(last_imu_data, curr_imu_data)

        eskf.SaveData(fuse_file, curr_imu_data)
        eskf.SaveGTData(gt_file, curr_imu_data)

        if is_obs_analysis:
            OA.SaveFGY(eskf.Fo, eskf.G, eskf.Y, curr_gnss_data.gnss_time)

    if is_obs_analysis:
        OA.ComputeSOM()
        OA.ComputeObservability()

    end_time = curr_imu_data.imu_time
    tick_end = time.time()
    fuse_file.close()
    gps_file.close()
    gt_file.close()
    # ins_file.close()
    print('fuse finished: {} imu data in {} seconds'.format(data_size, end_time-start_time))
    print('time cost: {:.2f} s'.format(tick_end-tick_start))

    
    # display
    display = True
    if display:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_title('compare path')
        if os.path.exists(fuse_file_name):
            fuse_data = np.loadtxt(fuse_file_name)
            ax.plot3D(fuse_data[:, 1], fuse_data[:, 2], fuse_data[:, 3], color='r', label='fuse_gps_imu')
        if os.path.exists(gps_file_name):
            gps_data = np.loadtxt(gps_file_name)
            ax.plot3D(gps_data[:, 1], gps_data[:, 2], gps_data[:, 3], color='g', alpha=0.5, label='gps')
        if os.path.exists(gt_file_name):
            gt_data = np.loadtxt(gt_file_name)
            ax.plot3D(gt_data[:, 1], gt_data[:, 2], gt_data[:, 3], color='b', label='ground_truth')
        #if os.path.exists(ins_file_name):
            #ins_data = np.loadtxt(ins_file_name)
            # ax.plot3D(ins_data[:, 1], ins_data[:, 2], ins_data[:, 3], color='y', label='ins')
        plt.legend(loc='best')
        plt.show()