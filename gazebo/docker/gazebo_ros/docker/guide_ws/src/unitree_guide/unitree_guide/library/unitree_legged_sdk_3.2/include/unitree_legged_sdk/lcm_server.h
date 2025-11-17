/************************************************************************
Copyright (c) 2020, Unitree Robotics.Co.Ltd. All rights reserved.
Use of this source code is governed by the MPL-2.0 license, see LICENSE.
************************************************************************/

#ifndef _UNITREE_LEGGED_LCM_SERVER_
#define _UNITREE_LEGGED_LCM_SERVER_

#include "comm.h"
#include "unitree_legged_sdk/unitree_legged_sdk.h"

namespace UNITREE_LEGGED_SDK
{
// Low command Lcm Server
class Lcm_Server_Low
{
public:
    Lcm_Server_Low(LeggedType rname) : udp(LOWLEVEL), mylcm(LOWLEVEL){
        udp.InitCmdData(cmd);
    }
    void UDPRecv(){
        udp.Recv();
    }
    void UDPSend(){
        udp.Send();
    }
    void LCMRecv();
    void RobotControl();

    UDP udp;
    LCM mylcm;
    LowCmd cmd = {0};
    LowState state = {0};
};
void Lcm_Server_Low::LCMRecv()
{
    if(mylcm.lowCmdLCMHandler.isrunning){
        pthread_mutex_lock(&mylcm.lowCmdLCMHandler.countMut);
        mylcm.lowCmdLCMHandler.counter++;
        if(mylcm.lowCmdLCMHandler.counter > 10){
            printf("Error! LCM Time out.\n");
            exit(-1);              // can be commented out
        }
        pthread_mutex_unlock(&mylcm.lowCmdLCMHandler.countMut);
    }
    mylcm.Recv();
}
void Lcm_Server_Low::RobotControl()
{
    udp.GetRecv(state);
    mylcm.Send(state);
    mylcm.Get(cmd);
    udp.SetSend(cmd);
}



// High command Lcm Server
class Lcm_Server_High
{
public:
    Lcm_Server_High(LeggedType rname): udp(HIGHLEVEL), mylcm(HIGHLEVEL){
        udp.InitCmdData(cmd);
    }
    void UDPRecv(){
        udp.Recv();
    }
    void UDPSend(){
        udp.Send();
    }
    void LCMRecv();
    void RobotControl();

    UDP udp;
    LCM mylcm;
    HighCmd cmd = {0};
    HighState state = {0};
};

void Lcm_Server_High::LCMRecv()
{
    if(mylcm.highCmdLCMHandler.isrunning){
        pthread_mutex_lock(&mylcm.highCmdLCMHandler.countMut);
        mylcm.highCmdLCMHandler.counter++;
        if(mylcm.highCmdLCMHandler.counter > 10){
            printf("Error! LCM Time out.\n");
            exit(-1);              // can be commented out
        }
        pthread_mutex_unlock(&mylcm.highCmdLCMHandler.countMut);
    }
    mylcm.Recv();
}

void Lcm_Server_High::RobotControl()
{
    udp.GetRecv(state);
    mylcm.Send(state);
    mylcm.Get(cmd);
    udp.SetSend(cmd);
}




}
#endif
