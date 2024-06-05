from scipy.spatial.transform import Rotation as R
import numpy as np
 
# euler q
def quaternion2euler(quaternion):
    r = R.from_quat(quaternion)
    euler = r.as_euler('xyz', degrees=True)
    return euler
 
def euler2quaternion(euler):
    r = R.from_euler('xyz', euler, degrees=True)
    quaternion = r.as_quat()
    return quaternion

# euler R
def euler2rm(euler):
    r = R.from_euler('xyz', euler, degrees=True)
    rotation_matrix = r.as_matrix()
    return rotation_matrix

def rm2euler(rotation_matrix):
    r = R.from_matrix(rotation_matrix)
    euler = r.as_euler('xyz', degrees=True)
    return euler

# q R
def quaternion2rm(quaternion:np.array):
    rotation_matrix = R.from_quat(quaternion).as_matrix()
    return rotation_matrix

def rm2quaternion(rotation_matrix:np.array):
    quaternion = R.from_matrix(rotation_matrix).as_quat()
    return quaternion

# rv q
def rv2quaternion(rotation_vector:np.array):
    rotation_vector_norm = np.linalg.norm(rotation_vector)
    rotation_vector_unit = rotation_vector/rotation_vector_norm
    quaternion = np.array([np.cos(rotation_vector_norm/2), np.sin(rotation_vector_norm/2)*rotation_vector_unit])
    return quaternion

def quaternion2rv(quaternion:np.array):
    quaternion_norm = np.linalg.norm(quaternion)
    quaternion_unit = quaternion/quaternion_norm
    rotation_vector = 2*np.arccos(quaternion_unit[0])*quaternion_unit[1:]
    return rotation_vector

# rv R  tested ok
def rv2rm(rotation_vector:np.array):
    rotation_vector_norm = np.linalg.norm(rotation_vector)
    if rotation_vector_norm == 0:
        return np.identity(3)
    else:
        rotation_vector_unit = rotation_vector/rotation_vector_norm
        rotation_matrix = np.identity(3) + BuildSkewSymmetricMatrix(rotation_vector_unit)*np.sin(rotation_vector_norm) + BuildSkewSymmetricMatrix(rotation_vector_unit)**2*(1-np.cos(rotation_vector_norm))
        return rotation_matrix

def BuildSkewSymmetricMatrix(vector:np.array):
    skew_symmetric_matrix = np.array([[0, -vector[2][0], vector[1][0]],
                                      [vector[2][0], 0, -vector[0][0]],
                                      [-vector[1][0], vector[0][0], 0]])
    return skew_symmetric_matrix

def angle_diff(angle1, angle2):
    d_angle = angle2 - angle1
    d_angle = d_angle % 360
    if d_angle > 180:
        d_angle -= 360
    elif d_angle < -180:
        d_angle += 360
    return d_angle

if __name__=='__main__':
    print(euler2rm(np.array([1.50689, -0.07204, 278.53459])))
    print(rm2quaternion(euler2rm(np.array([1.50689, -0.07204, 278.53459]))))
    print(euler2quaternion(np.array([1.50689, -0.07204, 278.53459])))
    print(quaternion2euler(euler2quaternion(np.array([1.50689, -0.07204, 278.53459]))))
    print(rv2rm(np.array([[1], [2], [3]])))

    from scipy.spatial.transform import Rotation


# 使用Rotation类从旋转矢量创建旋转矩阵
    rotation = Rotation.from_rotvec([1,2,3])
    rotation_matrix = rotation.as_matrix()
    print(rotation_matrix)