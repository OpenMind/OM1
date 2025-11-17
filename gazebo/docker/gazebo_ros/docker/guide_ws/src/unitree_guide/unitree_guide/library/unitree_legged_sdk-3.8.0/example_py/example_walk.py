#!/usr/bin/python

import sys
import time

sys.path.append("../lib/python/amd64")
import robot_interface as sdk

if __name__ == "__main__":

    HIGHLEVEL = 0xEE
    LOWLEVEL = 0xFF

    udp = sdk.UDP(HIGHLEVEL, 8080, "192.168.123.161", 8082)

    cmd = sdk.HighCmd()
    state = sdk.HighState()
    udp.InitCmdData(cmd)

    motiontime = 0
    while True:
        time.sleep(0.002)
        motiontime = motiontime + 1

        udp.Recv()
        udp.GetRecv(state)

        # print(motiontime)
        # print(state.imu.rpy[0])
        # print(motiontime, state.motorState[0].q, state.motorState[1].q, state.motorState[2].q)
        # print(state.imu.rpy[0])

        cmd.mode = (
            0  # 0:idle, default stand      1:forced stand     2:walk continuously
        )
        cmd.gaitType = 0
        cmd.speedLevel = 0
        cmd.footRaiseHeight = 0
        cmd.bodyHeight = 0
        cmd.euler = [0, 0, 0]
        cmd.velocity = [0, 0]
        cmd.yawSpeed = 0.0
        cmd.reserve = 0

        # cmd.mode = 2
        # cmd.gaitType = 1
        # # cmd.position = [1, 0]
        # # cmd.position[0] = 2
        # cmd.velocity = [-0.2, 0] # -1  ~ +1
        # cmd.yawSpeed = 0
        # cmd.bodyHeight = 0.1

        if motiontime > 0 and motiontime < 1000:
            cmd.mode = 1
            cmd.euler = [-0.3, 0, 0]

        if motiontime > 1000 and motiontime < 2000:
            cmd.mode = 1
            cmd.euler = [0.3, 0, 0]

        if motiontime > 2000 and motiontime < 3000:
            cmd.mode = 1
            cmd.euler = [0, -0.2, 0]

        if motiontime > 3000 and motiontime < 4000:
            cmd.mode = 1
            cmd.euler = [0, 0.2, 0]

        if motiontime > 4000 and motiontime < 5000:
            cmd.mode = 1
            cmd.euler = [0, 0, -0.2]

        if motiontime > 5000 and motiontime < 6000:
            cmd.mode = 1
            cmd.euler = [0.2, 0, 0]

        if motiontime > 6000 and motiontime < 7000:
            cmd.mode = 1
            cmd.bodyHeight = -0.2

        if motiontime > 7000 and motiontime < 8000:
            cmd.mode = 1
            cmd.bodyHeight = 0.1

        if motiontime > 8000 and motiontime < 9000:
            cmd.mode = 1
            cmd.bodyHeight = 0.0

        if motiontime > 9000 and motiontime < 11000:
            cmd.mode = 5

        if motiontime > 11000 and motiontime < 13000:
            cmd.mode = 6

        if motiontime > 13000 and motiontime < 14000:
            cmd.mode = 0

        if motiontime > 14000 and motiontime < 18000:
            cmd.mode = 2
            cmd.gaitType = 2
            cmd.velocity = [0.4, 0]  # -1  ~ +1
            cmd.yawSpeed = 2
            cmd.footRaiseHeight = 0.1
            # printf("walk\n")

        if motiontime > 18000 and motiontime < 20000:
            cmd.mode = 0
            cmd.velocity = [0, 0]

        if motiontime > 20000 and motiontime < 24000:
            cmd.mode = 2
            cmd.gaitType = 1
            cmd.velocity = [0.2, 0]  # -1  ~ +1
            cmd.bodyHeight = 0.1
            # printf("walk\n")

        udp.SetSend(cmd)
        udp.Send()
