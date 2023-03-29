import numpy as np
import math

def degree(x):
    pi=math.pi
    degree=(x*180)/pi
    return degree

def quaternion_to_euler_angle_vectorized2(w, x, y, z, rad=False):
    """
    https://stackoverflow.com/questions/56207448/efficient-quaternions-to-euler-transformation
    """
    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = np.degrees(np.arctan2(t0, t1))

    t2 = +2.0 * (w * y - z * x)

    t2 = np.clip(t2, a_min=-1.0, a_max=1.0)
    Y = np.degrees(np.arcsin(t2))

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = np.degrees(np.arctan2(t3, t4))

    if rad:
        return np.radians(X), np.radians(Y), np.radians(Z)
    return X, Y, Z

def euler_to_quaternion(roll, pitch, yaw):
    """
    # https://computergraphics.stackexchange.com/questions/8195/how-to-convert-euler-angles-to-quaternions-and-get-the-same-euler-angles-back-fr
    Args:
        roll: radians
        pitch: radians
        yaw: radians

    Returns:

    """
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(
        yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(
        yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(
        yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(
        yaw / 2)

    return [qx, qy, qz, qw]

if __name__=='__main__':
    # print(quaternion_to_euler_angle_vectorized2(0.707388268792, 1.13872376639e-07, -2.07007633433e-06, -0.706825181477))
    print(quaternion_to_euler_angle_vectorized2(0.999999999997,2.56961380174e-06, -3.30265582406e-07, -1.07989486604e-08))
