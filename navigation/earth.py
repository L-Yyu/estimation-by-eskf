import numpy as np
import math

# WGS84椭球参数
W = 7.2921151467e-5 # 地球自转角速度
A = 6378137.0 # 半长轴
B = 6356752.3142451793 # 半短轴
F = 0.0033528106647474805 # 扁率 
E1_SQ = 0.0066943799901413156 # 第一偏心率平方
 
def geodetic_to_ecef(lat, lon, h):
    # (lat, lon) in degrees
    # h in meters
    lamb = math.radians(lat)
    phi = math.radians(lon)
    s = math.sin(lamb)
    N = A / math.sqrt(1 - E1_SQ * s * s)
 
    sin_lambda = math.sin(lamb)
    cos_lambda = math.cos(lamb)
    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)
 
    x = (h + N) * cos_lambda * cos_phi
    y = (h + N) * cos_lambda * sin_phi
    z = (h + (1 - E1_SQ) * N) * sin_lambda
 
    return x, y, z
 
def ecef_to_enu(x, y, z, lat0, lon0, h0):
    lamb = math.radians(lat0)
    phi = math.radians(lon0)
    s = math.sin(lamb)
    N = A / math.sqrt(1 - E1_SQ * s * s)
 
    sin_lambda = math.sin(lamb)
    cos_lambda = math.cos(lamb)
    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)
 
    x0 = (h0 + N) * cos_lambda * cos_phi
    y0 = (h0 + N) * cos_lambda * sin_phi
    z0 = (h0 + (1 - E1_SQ) * N) * sin_lambda
 
    xd = x - x0
    yd = y - y0
    zd = z - z0
 
    t = -cos_phi * xd -  sin_phi * yd
 
    xEast = -sin_phi * xd + cos_phi * yd
    yNorth = t * sin_lambda  + cos_lambda * zd
    zUp = cos_lambda * cos_phi * xd + cos_lambda * sin_phi * yd + sin_lambda * zd
 
    return xEast, yNorth, zUp
 
def enu_to_ecef(xEast, yNorth, zUp, lat0, lon0, h0):
    lamb = math.radians(lat0)
    phi = math.radians(lon0)
    s = math.sin(lamb)
    N = A / math.sqrt(1 - E1_SQ * s * s)
 
    sin_lambda = math.sin(lamb)
    cos_lambda = math.cos(lamb)
    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)
 
    x0 = (h0 + N) * cos_lambda * cos_phi
    y0 = (h0 + N) * cos_lambda * sin_phi
    z0 = (h0 + (1 - E1_SQ) * N) * sin_lambda
 
    t = cos_lambda * zUp - sin_lambda * yNorth
 
    zd = sin_lambda * zUp + cos_lambda * yNorth
    xd = cos_phi * t - sin_phi * xEast 
    yd = sin_phi * t + cos_phi * xEast
 
    x = xd + x0 
    y = yd + y0 
    z = zd + z0 
 
    return x, y, z
 
def ecef_to_geodetic(x, y, z):
   # Convert from ECEF cartesian coordinates to 
   # latitude, longitude and height.  WGS-84
    x2 = x ** 2 
    y2 = y ** 2 
    z2 = z ** 2 
 
    a = 6378137.0000    # earth radius in meters
    b = 6356752.3142    # earth semiminor in meters 
    e = math.sqrt (1-(b/a)**2) 
    b2 = b*b 
    e2 = e ** 2 
    ep = e*(a/b) 
    r = math.sqrt(x2+y2) 
    r2 = r*r 
    E2 = a ** 2 - b ** 2 
    F = 54*b2*z2 
    G = r2 + (1-e2)*z2 - e2*E2 
    c = (e2*e2*F*r2)/(G*G*G) 
    s = ( 1 + c + math.sqrt(c*c + 2*c) )**(1/3) 
    P = F / (3 * (s+1/s+1)**2 * G*G) 
    Q = math.sqrt(1+2*e2*e2*P) 
    ro = -(P*e2*r)/(1+Q) + math.sqrt((a*a/2)*(1+1/Q) - (P*(1-e2)*z2)/(Q*(1+Q)) - P*r2/2) 
    tmp = (r - e2*ro) ** 2 
    U = math.sqrt( tmp + z2 ) 
    V = math.sqrt( tmp + (1-e2)*z2 ) 
    zo = (b2*z)/(a*V) 
 
    height = U*( 1 - b2/(a*V) ) 
    
    lat = math.atan( (z + ep*ep*zo)/r ) 
 
    temp = math.atan(y/x) 
    if x >=0 :    
        long = temp 
    elif (x < 0) & (y >= 0):
        long = math.pi + temp 
    else :
        long = temp - math.pi 
 
    lat0 = lat/(math.pi/180) 
    lon0 = long/(math.pi/180) 
    h0 = height 
 
    return lat0, lon0, h0
 
 
def geodetic_to_enu(lat, lon, h, lat_ref, lon_ref, h_ref):
 
    x, y, z = geodetic_to_ecef(lat, lon, h)
    
    return ecef_to_enu(x, y, z, lat_ref, lon_ref, h_ref)
 
def enu_to_geodetic(xEast, yNorth, zUp, lat_ref, lon_ref, h_ref):
 
    x,y,z = enu_to_ecef(xEast, yNorth, zUp, lat_ref, lon_ref, h_ref)
 
    return ecef_to_geodetic(x,y,z)


def lla2ned(lla, ref_lla):
    x, y, z = geodetic_to_enu(lla[0][0], lla[1][0], lla[2][0], ref_lla[0][0], ref_lla[1][0], ref_lla[2][0])
    return np.array([[y],[x],[-z]])
def ned2lla(ned, ref_lla):
    lat, lon, h = enu_to_geodetic(ned[1][0], ned[0][0], -ned[2][0], ref_lla[0][0], ref_lla[1][0], ref_lla[2][0])
    return np.array([[lat],[lon],[h]])

# n系(导航坐标系)到e系(地心地固坐标系)转换矩阵
def cne(lla):
    lat = lla[0][0] # 纬度
    lon = lla[1][0] # 经度
    h = lla[2][0]
    sinlat = math.sin(lat)
    sinlon = math.sin(lon)
    coslat = math.cos(lat)
    coslon = math.cos(lon)
    dcm = np.array([[-sinlat * coslon, -sinlon, -coslat * coslon],
                    [-sinlat * sinlon, coslon, -coslat * sinlon],
                    [coslat, 0, -sinlat]])
    return dcm

# 计算重力加速度
def GetGravity(lla):
    lat = lla[0][0] # 纬度 deg
    h = lla[2][0]
    lat_rad = math.radians(lat)
    sin_lat_sq = math.sin(lat_rad) ** 2
    return 9.7803267715 * (1 + 0.0052790414 * sin_lat_sq + 0.0000232718 * sin_lat_sq ** 2) + \
           h * (0.0000000043977311 * sin_lat_sq - 0.0000030876910891) + 0.0000000000007211 * h ** 2

def GetWie_n(lla):
    lat = lla[0][0] # 纬度 deg
    lat_rad = math.radians(lat)
    wie_n =  np.array([W * math.cos(lat_rad), 0, -W * math.sin(lat_rad)]).reshape(3,1)
    return wie_n

def GetWen_n(lla, v_n):
    lat = lla[0][0] # 纬度 deg
    lat_rad = math.radians(lat)
    height = lla[2][0]
    rn = A / math.sqrt(1 - E1_SQ * math.sin(lat_rad) ** 2) # 子午圈曲率半径
    rm = A * (1 - E1_SQ) / math.pow(1 - E1_SQ * math.sin(lat_rad) ** 2, 1.5) # 卯酉圈曲率半径
    wen_n = np.array([v_n[1][0] / (rm + height), -v_n[0][0] / (rn + height), -v_n[1][0] / (rn + height) * math.tan(lat_rad)]).reshape(3,1)
    return wen_n


if __name__=='__main__':
    lla = np.array([[30.4447873701], [114.4718632047], [20.899]])
    lla1 = np.array([[30.4447858449], [114.4718661417], [21.093]])
    ned = lla2ned(lla1, lla)
    wie_n = GetWie_n(lla)
    wen_n = GetWen_n(lla, np.array([[20],[20],[10]]))
    print(wie_n)
    print(wen_n)
  