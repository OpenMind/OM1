/*****************************************************************
 Copyright (c) 2020, Unitree Robotics.Co.Ltd. All rights reserved.
******************************************************************/

#include "unitree_legged_sdk/unitree_legged_sdk.h"
#include <math.h>
#include <iostream>

using namespace UNITREE_LEGGED_SDK;

class Custom
{
public:
    Custom(uint8_t level):
        safe(LeggedType::Go1),
        udp(level, 8090, "192.168.123.10", 8007){
        udp.InitCmdData(cmd);
    }
    void UDPRecv();
    void UDPSend();
    void RobotControl();

    Safety safe;
    UDP udp;
    LowCmd cmd = {0};
    LowState state = {0};
    int Tpi = 0;
    int motiontime = 0;
    float dt = 0.002;     // 0.001~0.01
};

void Custom::UDPRecv()
{
    udp.Recv();
}

void Custom::UDPSend()
{
    udp.Send();
}

void Custom::RobotControl()
{
    motiontime++;
    udp.GetRecv(state);

    // gravity compensation
    cmd.motorCmd[FR_0].tau = -0.65f;
    cmd.motorCmd[FL_0].tau = +0.65f;
    cmd.motorCmd[RR_0].tau = -0.65f;
    cmd.motorCmd[RL_0].tau = +0.65f;

    if( motiontime >= 500){
        float speed = 2 * sin(3*M_PI*Tpi/1500.0);

        cmd.motorCmd[FR_1].q = PosStopF;
        cmd.motorCmd[FR_1].dq = speed;
        cmd.motorCmd[FR_1].Kp = 0;
        cmd.motorCmd[FR_1].Kd = 4;
        // cmd.motorCmd[FR_1].Kd = 6;
        cmd.motorCmd[FR_1].tau = 0.0f;
        Tpi++;
    }

    if(motiontime > 10){
        int res1 = safe.PowerProtect(cmd, state, 1);
        // You can uncomment it for position protection
        // int res2 = safe.PositionProtect(cmd, state, 10);
        if(res1 < 0) exit(-1);
    }

    udp.SetSend(cmd);
}

int main(void)
{
    std::cout << "Communication level is set to LOW-level." << std::endl
              << "WARNING: Make sure the robot is hung up." << std::endl
              << "Press Enter to continue..." << std::endl;
    std::cin.ignore();

    Custom custom(LOWLEVEL);
    // InitEnvironment();
    LoopFunc loop_control("control_loop", custom.dt,    boost::bind(&Custom::RobotControl, &custom));
    LoopFunc loop_udpSend("udp_send",     custom.dt, 3, boost::bind(&Custom::UDPSend,      &custom));
    LoopFunc loop_udpRecv("udp_recv",     custom.dt, 3, boost::bind(&Custom::UDPRecv,      &custom));

    loop_udpSend.start();
    loop_udpRecv.start();
    loop_control.start();

    while(1){
        sleep(10);
    };

    return 0;
}
