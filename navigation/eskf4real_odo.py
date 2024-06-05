import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import queue
import os
import yaml
from data4real import IMUData, GNSSData, ODOData
from tools import euler2rm, euler2quaternion, rm2quaternion, rv2rm, BuildSkewSymmetricMatrix
from earth import lla2ned, ned2lla, GetGravity, GetWie_n, GetWen_n
from observability import ObservabilityAnalysis
from tqdm import tqdm
import time

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
        # 15 6  15 12   21 6 21 18
        self.dim_state = states_rank        #   15:p v phi bg ba    21:p v phi bg ba sg sa
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
        self.antenna_lever_ = np.array(config['ant_lever']).reshape(3,1)
        self.is_earth_rotation = config['is_earth_rotation']  # 是否考虑地球自转
        # 初始化状态
        self.InitState(config)
        

    def SetQ(self, arw, vrw, bg_std, ba_std, sg_std, sa_std, corr_time):
        # 设置imu噪声参数
        # 转换为标准单位
        arw = np.array(arw) * D2R/60.0   # deg/sqrt(h) -> rad/sqrt(s)
        vrw = np.array(vrw) / 60.0  # m/s/sqrt(h) -> m/s/sqrt(s)
        bg_std = np.array(bg_std) * D2R/ 3600.0  # deg/h -> rad/s
        ba_std = np.array(ba_std) * 1e-5    # mGal -> m/s^2
        sg_std = np.array(sg_std) * 1e-6    # ppm -> 1
        sa_std = np.array(sa_std) * 1e-6    # ppm -> 1
        corr_time = np.array(corr_time) * 3600.0  # h -> s
        # self.Q = np.zeros((self.dim_state_noise, self.dim_state_noise))
        if self.dim_state == 15 and self.dim_state_noise == 6:
            self.Q[0:3, 0:3] = np.eye(3) * arw * arw
            self.Q[3:6, 3:6] = np.eye(3) * vrw * vrw
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


    def Predict(self, last_imu_data:IMUData, curr_imu_data:IMUData):
        delta_t = curr_imu_data.imu_time - last_imu_data.imu_time
        # 根据上一时刻状态，计算状态转移矩阵F_k-1, B_k-1
        self.UpdateErrorState(delta_t, last_imu_data)

        # imu数据补偿
        curr_imu_data.imu_angle_increment -= self.gyro_bias_ * delta_t  # rad
        curr_imu_data.imu_vel_increment -= self.accel_bias_ * delta_t  # m/s
        if self.dim_state == 21:
            curr_imu_data.imu_angle_increment = (np.ones((3,1)) + self.gyro_scale_) * curr_imu_data.imu_angle_increment
            curr_imu_data.imu_vel_increment = (np.ones((3,1)) + self.accel_scale_) * curr_imu_data.imu_vel_increment
        # 姿态更新 基于旋转矩阵的更新算法
        last_Cb_n = self.rotation_matrix_
        # 计算Cbb
        rotation_vector = curr_imu_data.imu_angle_increment + 1/12 * BuildSkewSymmetricMatrix(last_imu_data.imu_angle_increment) @ curr_imu_data.imu_angle_increment # rad
        Cb_b = rv2rm(rotation_vector)
        # 计算Cnn：由地球自转和n系变化的影响
        if self.is_earth_rotation:
            wie_n = GetWie_n(self.pos_lla_)
        else:
            wie_n = np.zeros((3,1))
        wen_n = GetWen_n(self.pos_lla_, self.velocity_)
        Cn_n = rv2rm(-(wie_n + wen_n) * delta_t)
        #if np.linalg.norm(-(wie_n + wen_n) * delta_t) > np.linalg.norm(rotation_vector): 
        #    print('earth rotation matters!')
        # 更新
        curr_Cb_n = Cn_n @ last_Cb_n @ Cb_b
        self.rotation_matrix_ = curr_Cb_n
        self.quat_ = rm2quaternion(curr_Cb_n)
    
        # 速度更新
        last_vel_n = self.velocity_
        curr_vel_n = last_vel_n + self.rotation_matrix_ @ curr_imu_data.imu_vel_increment + self.g_n * delta_t
        self.velocity_ = curr_vel_n

        # 位置更新
        self.pos_ = self.pos_ + 0.5 * delta_t * (curr_vel_n + last_vel_n)
        self.pos_lla_ = ned2lla(self.pos_, self.ref_pos_lla_)
        # self.g_ = np.array([0, 0, -GetGravity(self.pos_lla_)]).reshape((3,1))
        
        if False:
            print(curr_imu_data.imu_time)
            #print(self.pos_)
            #print(self.pos_lla_)
            #print(self.velocity_)
            #print(rm2euler(self.rotation_matrix_))
            pass

    def UpdateErrorState(self, delta_t, imu_data):
        accel_b = imu_data.imu_vel_increment / delta_t 
        accel_n = self.rotation_matrix_ @ accel_b
        w_ib_b = imu_data.imu_angle_increment / delta_t
        w_ib_n = self.rotation_matrix_ @ w_ib_b

        # 状态转移矩阵
        F_23 = BuildSkewSymmetricMatrix(accel_n)
        if self.is_earth_rotation:
            w_ie_n = GetWie_n(self.pos_lla_)
        else:
            w_ie_n = np.zeros((3,1))
        w_en_n = GetWen_n(self.pos_lla_, self.velocity_)
        F_33 = -BuildSkewSymmetricMatrix(w_ie_n + w_en_n)

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

        # 离散化1
        #Phi = np.eye(self.dim_state) + self.F * delta_t
        #Bk = self.B * delta_t
        #Qd = Bk @ self.Q @ Bk.T
        # 用于可观测性分析
        # self.Fo = self.F * delta_t
        
        # 离散化2
        Phi = np.eye(self.dim_state) + self.F * delta_t
        Qd = self.B @ self.Q @ self.B.T * delta_t
        Qd = (Phi @ Qd @ Phi.T + Qd) / 2
        # 用于可观测性分析
        self.Fo = Phi

        # KF prediction
        self.X = Phi @ self.X
        self.P = Phi @ self.P @ Phi.T + Qd


    def Correct(self, curr_gnss_data:GNSSData):
        self.G[ID_MEASUREMENT_P:ID_MEASUREMENT_P+3, ID_STATE_P:ID_STATE_P+3] = np.eye(3)
        self.G[ID_MEASUREMENT_P:ID_MEASUREMENT_P+3, ID_STATE_PHI:ID_STATE_PHI+3] = BuildSkewSymmetricMatrix(self.rotation_matrix_ @ self.antenna_lever_)

        atn_pos_predict = self.rotation_matrix_ @ self.antenna_lever_ + self.pos_
        self.Y = atn_pos_predict - lla2ned(curr_gnss_data.position_lla, self.ref_pos_lla_)
        #self.Y = self.pos_ - lla2ned(curr_gnss_data.position_lla, self.ref_pos_lla_)
        self.K = self.P @ self.G.T @ np.linalg.inv(self.G @ self.P @ self.G.T + self.C @ self.R @ self.C.T)
        self.P = (np.eye(self.dim_state) - self.K @ self.G) @ self.P
        self.X = self.X + self.K @ (self.Y - self.G @ self.X)

        # 误差状态反馈至完整状态
        self.StateFeedback()
        self.pos_lla_ = ned2lla(self.pos_, self.ref_pos_lla_)

        # 误差状态置零
        self.ResetState()

    def CorrectVn(self, curr_odo_data):
        self.G1 = np.zeros((3, self.dim_state))
        self.G1[0:3, ID_STATE_V:ID_STATE_V+3] = np.eye(3)
        self.C1 = np.identity(3)
        self.R1 = np.zeros((3,3))
        self.R1[0,0] = 0.1 * 0.1
        self.R1[1,1] = 0.01
        self.R1[2,2] = 0.01

        self.Y1 = self.velocity_ - curr_odo_data.v_n 
        self.K1 = self.P @ self.G1.T @ np.linalg.inv(self.G1 @ self.P @ self.G1.T + self.C1 @ self.R1 @ self.C1.T)
        self.P = (np.eye(self.dim_state) - self.K1 @ self.G1) @ self.P
        self.X = self.X + self.K1 @ (self.Y1 - self.G1 @ self.X)
        
        self.StateFeedback()
        self.pos_lla_ = ned2lla(self.pos_, self.ref_pos_lla_)

        # 误差状态置零
        self.ResetState()

    def CorrectVb(self, curr_odo_data):
        self.G1 = np.zeros((3, self.dim_state))
        self.G1[0:3, ID_STATE_V:ID_STATE_V+3] = self.rotation_matrix_.T
        self.G1[0:3, ID_STATE_PHI:ID_STATE_PHI+3] = -self.rotation_matrix_.T @ BuildSkewSymmetricMatrix(self.velocity_)
        self.C1 = np.identity(3)
        self.R1 = np.zeros((3,3))
        self.R1[0,0] = 0.01
        self.R1[1,1] = 10
        self.R1[2,2] = 5

        #self.Y1 = self.rotation_matrix_.T @ self.velocity_ - curr_odo_data.v_b
        self.Y1 = self.rotation_matrix_.T @ self.velocity_ - np.array([curr_odo_data.v_b[0][0],0,0]).reshape(3,1)
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
        self.gyro_bias_ = self.gyro_bias_ + self.X[ID_STATE_BG:ID_STATE_BG+3, :]
        self.accel_bias_ = self.accel_bias_ + self.X[ID_STATE_BA:ID_STATE_BA+3, :]
        if self.dim_state == 21:
            self.gyro_scale_ = self.gyro_scale_ + self.X[ID_STATE_SG:ID_STATE_SG+3, :]
            self.accel_scale_ = self.accel_scale_ + self.X[ID_STATE_SA:ID_STATE_SA+3, :]


    def ResetState(self):
        self.X = np.zeros((self.dim_state, 1))

    def SaveData(self, file, imu_data:IMUData):
        file.write(str(imu_data.imu_time)+' ')
        for i in range(3):
            file.write(str(eskf.pos_[i][0])+' ')
        for i in range(4):
            if i < 3:
                file.write(str(eskf.quat_[i])+' ')
            else:
                file.write(str(eskf.quat_[i])+'\n')

    def SaveGnssData(self, file, curr_gnss_data:GNSSData):
        curr_position_ned = lla2ned(curr_gnss_data.position_lla, self.ref_pos_lla_)
        file.write(str(curr_gnss_data.gnss_time)+' ')
        for i in range(3):
            file.write(str(curr_position_ned[i][0])+' ')
        file.write('0 0 0 1\n')
        
if __name__ == "__main__":
    data_path = './data/i300'
    states_rank = 21
    noise_rank = 18
    is_obs_analysis = False
    is_odo = False
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
    ins_file_name = os.path.join(data_path,'ins.txt')
    gins_data_path = os.path.join(data_path,'fused_kfgins.txt')
    
    fuse_file = open(fuse_file_name,'w')
    gps_file = open(gps_file_name,'w')
    # gt_file = open(gt_file_name,'w')
    # ins_file = open(ins_file_name,'w')
    
    # imu gps fuse
    tick_start = time.time()
    eskf = ESKF(config, states_rank, noise_rank)

    imu_data_queue = IMUData.read_imu_data(os.path.join(data_path,'imu.txt'))
    gnss_data_queue = GNSSData.read_gnss_data(os.path.join(data_path,'GNSS-RTK.txt'))
    odo_data_queue = ODOData.read_odo_data(os.path.join(data_path,'odo.txt'))
    curr_imu_data = imu_data_queue.get()
    curr_gnss_data = gnss_data_queue.get()
    curr_odo_data = odo_data_queue.get()

    start_time = config['start_time']
    end_time = config['end_time']
    gt_start_index = 0  # 用于显示对比
    while(curr_imu_data.imu_time <= start_time):
        curr_imu_data = imu_data_queue.get()
        gt_start_index += 1
    while(curr_gnss_data.gnss_time <= start_time or curr_gnss_data.gnss_time < curr_imu_data.imu_time):
        curr_gnss_data = gnss_data_queue.get()
    while(curr_odo_data.odo_time <= start_time):
        curr_odo_data = odo_data_queue.get()

    data_size = (end_time - start_time) * 200
    for i in tqdm(range(data_size), desc="fusing IMU and GNSS"):
        if imu_data_queue.empty() or gnss_data_queue.empty():
            break
        last_imu_data = curr_imu_data
        curr_imu_data = imu_data_queue.get()
        curr_odo_data = odo_data_queue.get()

        if abs(curr_gnss_data.gnss_time - last_imu_data.imu_time) < 0.001:
            # 靠近上一时刻
            eskf.Correct(curr_gnss_data)
            eskf.SaveGnssData(gps_file, curr_gnss_data)
            curr_gnss_data = gnss_data_queue.get()

            eskf.Predict(last_imu_data, curr_imu_data)
        elif abs(curr_gnss_data.gnss_time - curr_imu_data.imu_time) < 0.001:
            # 靠近当前时刻
            eskf.Predict(last_imu_data, curr_imu_data)

            eskf.Correct(curr_gnss_data)
            eskf.SaveGnssData(gps_file, curr_gnss_data)
            curr_gnss_data = gnss_data_queue.get()
        elif (curr_gnss_data.gnss_time > last_imu_data.imu_time) and (curr_gnss_data.gnss_time < curr_imu_data.imu_time):
            # 内插到gnss时刻
            mid_imu_data = IMUData.Interpolate(last_imu_data, curr_imu_data, curr_gnss_data.gnss_time)
            eskf.Predict(last_imu_data, mid_imu_data)

            eskf.Correct(curr_gnss_data)
            eskf.SaveGnssData(gps_file, curr_gnss_data)
            curr_gnss_data = gnss_data_queue.get()

            eskf.Predict(mid_imu_data, curr_imu_data)
        else:
            eskf.Predict(last_imu_data, curr_imu_data)

        # odo 数据200hz，但是整秒处有额外数据。只在整秒处进行矫正
        if is_odo:
            if curr_odo_data.odo_time - curr_imu_data.imu_time < 0.001:
                # print(curr_odo_data.odo_time, curr_imu_data.imu_time)
                # print('odo correct')
                eskf.CorrectVb(curr_odo_data)
                curr_odo_data = odo_data_queue.get()
            elif curr_odo_data.odo_time < curr_imu_data.imu_time:
                curr_odo_data = odo_data_queue.get()

        eskf.SaveData(fuse_file, curr_imu_data)

        if is_obs_analysis:
            OA.SaveFGY(eskf.F, eskf.G, eskf.Y, curr_gnss_data.gnss_time)
    
    if is_obs_analysis:
        OA.ComputeSOM()
        OA.ComputeObservability()

    tick_end = time.time()
    fuse_file.close()
    gps_file.close()
    # gt_file.close()
    # ins_file.close()
    print('fuse finished: {} imu data in {} seconds'.format(data_size, end_time-start_time))
    print('time cost: {:.2f} s'.format(tick_end-tick_start))

    # display
    print(gt_start_index, data_size)
    display = True
    if display:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_title('compare path')
        if os.path.exists(fuse_file_name):
            fuse_data = np.loadtxt(fuse_file_name)
            ax.plot3D(fuse_data[:data_size, 1], fuse_data[:data_size, 2], fuse_data[:data_size, 3], color='r', label='fuse_gps_imu')
        if os.path.exists(gps_file_name):
            gps_data = np.loadtxt(gps_file_name)
            ax.plot3D(gps_data[:data_size//200, 1], gps_data[:data_size//200, 2], gps_data[:data_size//200, 3], color='g', alpha=0.5, label='gps')
        if os.path.exists(gt_file_name):
            gt_data = np.loadtxt(gt_file_name)
            ax.plot3D(gt_data[gt_start_index:gt_start_index+data_size, 1], gt_data[gt_start_index:gt_start_index+data_size, 2], gt_data[gt_start_index:gt_start_index+data_size, 3], color='b', label='ground_truth')
        if os.path.exists(ins_file_name):
            ins_data = np.loadtxt(ins_file_name)
            # ax.plot3D(ins_data[:, 1], ins_data[:, 2], ins_data[:, 3], color='y', label='ins')
        if os.path.exists(gins_data_path):
            gins_data = np.loadtxt(gins_data_path)
            ax.plot3D(gins_data[:data_size, 1], gins_data[:data_size, 2], gins_data[:data_size, 3], color='c', label='gins')
        plt.legend(loc='best')
        plt.show()