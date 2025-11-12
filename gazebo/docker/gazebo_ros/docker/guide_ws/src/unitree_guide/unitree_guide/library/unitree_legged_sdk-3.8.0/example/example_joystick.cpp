/************************************************************************
Copyright (c) 2020, Unitree Robotics.Co.Ltd. All rights reserved.
Use of this source code is governed by the MPL-2.0 license, see LICENSE.
************************************************************************/

#include "unitree_legged_sdk/unitree_legged_sdk.h"
#include "unitree_legged_sdk/joystick.h"
#include <math.h>
#include <iostream>
#include <unistd.h>

using namespace UNITREE_LEGGED_SDK;

class Custom
{
public:
    Custom(uint8_t level):
        safe(LeggedType::Go1),
        udp(level, 8090, "192.168.123.10", 8007){
        udp.InitCmdData(cmd);
    }
    void UDPSend();
    void UDPRecv();
    void RobotControl();

    Safety safe;
    UDP udp;
    LowCmd cmd = {0};
    LowState state = {0};
    xRockerBtnDataStruct _keyData;
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

    memcpy(&_keyData, &state.wirelessRemote[0], 40);

    if((int)_keyData.btn.components.A == 1){
        std::cout << "The key A is pressed, and the value of lx is " << _keyData.lx << std::endl;
    }

    safe.PowerProtect(cmd, state, 1);
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
