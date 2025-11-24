#!/usr/bin/env python3
import os
import sys
import time
from ctypes import *

from typedef import *

system = platform.system()
if system == "Windows":
    fd = c.open_set(b"\\\\.\\COM4")
    libPath = os.path.dirname(os.getcwd()) + "/lib/libUnitree_motor_SDK_Win64.dll"
elif system == "Linux":
    fd = c.open_set(b"/dev/ttyUSB0")
    maxbit = sys.maxsize
    if maxbit > 2**32:
        libPath = os.path.dirname(os.getcwd()) + "/lib/libUnitree_motor_SDK_Linux64.so"
        print("Linux 64 bits")
    else:
        libPath = os.path.dirname(os.getcwd()) + "/lib/libUnitree_motor_SDK_Linux32.so"
        print("Linux 32 bits")

c = cdll.LoadLibrary(libPath)

motor_s = MOTOR_send()
motor_s1 = MOTOR_send()
motor_r = MOTOR_recv()

motor_s.id = 0
motor_s.mode = 10
motor_s.T = 0.05  # 单位：Nm, T<255.9
motor_s.W = 0.0  # 单位：rad/s, W<511.9
motor_s.Pos = 0  # 单位：rad, Pos<131071.9
motor_s.K_P = 0.0  # K_P<31.9
motor_s.K_W = 0.0  # K_W<63.9

motor_s1.id = 0
motor_s1.mode = 0

c.modify_data(byref(motor_s))
c.modify_data(byref(motor_s1))

print("START")

i = 0
while i < 5:
    c.send_recv(fd, byref(motor_s), byref(motor_r))
    c.extract_data(byref(motor_r))
    print("*******************")
    print("Motor torque: ", motor_r.T)
    print("Motor position: ", motor_r.Pos)
    print("Motor velocity: ", motor_r.W)
    time.sleep(1)
    i = i + 1

c.send_recv(fd, byref(motor_s1), byref(motor_r))
print("END")

c.close_serial(fd)
