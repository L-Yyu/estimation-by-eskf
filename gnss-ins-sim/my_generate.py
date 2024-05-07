# -*- coding: utf-8 -*-
# Filename: demo_no_algo.py


import numpy as np
import os
import math
from gnss_ins_sim.sim import imu_model
from gnss_ins_sim.sim import ins_sim

# globals
D2R = math.pi/180

fs = 100.0          # IMU sample frequency
fs_gps = 10.0       # GPS sample frequency
fs_mag = fs         # magnetometer sample frequency, not used for now

def test_path_gen():
    '''
    test only path generation in Sim.
    '''
    # mid accuracy, partly from IMU381
    """
    gyro_mid_accuracy = {'b': np.array([0.0, 0.0, 0.0]) * D2R,
                         'b_drift': np.array([3.5, 3.5, 3.5]) * D2R/3600.0,
                         'b_corr':np.array([100.0, 100.0, 100.0]),
                         'arw': np.array([0.25, 0.25, 0.25]) * D2R/60}
    accel_mid_accuracy = {'b': np.array([0.0e-3, 0.0e-3, 0.0e-3]),
                          'b_drift': np.array([5.0e-5, 5.0e-5, 5.0e-5]),
                          'b_corr': np.array([100.0, 100.0, 100.0]),
                          'vrw': np.array([0.03, 0.03, 0.03]) / 60}
    'gyro_b': gyro bias, deg/hr
    'gyro_arw': gyro angle random walk, deg/rt-hr
    'gyro_b_stability': gyro bias instability, deg/hr
    'gyro_b_corr': gyro bias isntability correlation time, sec
    'accel_b': accel bias, m/s2
    'accel_vrw' : accel velocity random walk, m/s/rt-hr
    'accel_b_stability': accel bias instability, m/s2
    'accel_b_corr': accel bias isntability correlation time, sec

    gps_low_accuracy = {'stdp': np.array([5.0, 5.0, 7.0]),
                    'stdv': np.array([0.05, 0.05, 0.05])}

    odo_low_accuracy = {'scale': 0.99,
                      'stdv': 0.1}
    """
    imu_err = {'gyro_b': np.array([0.0, 0.0, 0.0]),
           'gyro_arw': np.array([0.25, 0.25, 0.25]) * 1.0,
           'gyro_b_stability': np.array([3.5, 3.5, 3.5]) * 1.0,
           'gyro_b_corr': np.array([100.0, 100.0, 100.0]),
           'accel_b': np.array([0.0e-3, 0.0e-3, 0.0e-3]),
           'accel_vrw': np.array([0.03, 0.03, 0.03]) * 1.0,
           'accel_b_stability': np.array([5.0e-5, 5.0e-5, 5.0e-5]) * 1.0,
           'accel_b_corr': np.array([100.0, 100.0, 100.0]),
           'mag_std': np.array([0.2, 0.2, 0.2]) * 1.0
          }
    gps_accuracy = {'stdp': np.array([0.01, 0.01, 0.02]),
                    'stdv': np.array([0.05, 0.05, 0.05])}
    odo_accuracy = {'scale': 0.99,
                    'stdv': 0.1}
    
    data_name = 'straight'

    # generate GPS data
    #imu = imu_model.IMU(accuracy=imu_err, axis=6, gps=True, gps_opt=gps_accuracy, odo_opt=odo_accuracy, odo=True)
    imu = imu_model.IMU(accuracy=imu_err, axis=6, gps=True, odo=True)

    #### start simulation
    sim = ins_sim.Sim([fs, fs_gps, fs_mag],
                      './motion_def_files'+'/'+data_name+'.csv',
                      ref_frame=0, # 0: NED
                      imu=imu,
                      mode=None,
                      env=None,
                      algorithm=None)
    sim.run(1)
    # save simulation data to files
    sim.results('./data'+'/'+data_name)
    # plot data, 3d plot of reference positoin, 2d plots of gyro and accel
    sim.plot(['ref_pos','accel', 'gyro', 'gps_visibility'], opt={'ref_pos': '3d'})

if __name__ == '__main__':
    test_path_gen()
