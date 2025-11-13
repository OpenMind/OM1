/**********************************************************************
 Copyright (c) 2020-2023, Unitree Robotics.Co.Ltd. All rights reserved.
***********************************************************************/
#ifndef CMDPANEL_H
#define CMDPANEL_H

#include "message/unitree_joystick.h"
#include "common/enumClass.h"
#include <pthread.h>

#ifdef COMPILE_WITH_REAL_ROBOT
    #ifdef ROBOT_TYPE_A1
        #include "unitree_legged_sdk/unitree_legged_sdk.h"
    #endif  // ROBOT_TYPE_A1
    #ifdef ROBOT_TYPE_Go1
        #include "unitree_legged_sdk/unitree_legged_sdk.h"
    #endif  // ROBOT_TYPE_Go1
#endif  // COMPILE_WITH_REAL_ROBOT

struct UserValue{
    float lx;
    float ly;
    float rx;
    float ry;
    float L2;
    UserValue(){
        setZero();
    }
    void setZero(){
        lx = 0;
        ly = 0;
        rx = 0;
        ry = 0;
        L2 = 0;
    }
};

class CmdPanel{
public:
    CmdPanel(){}
    virtual ~CmdPanel(){}
    UserCommand getUserCmd(){return userCmd;}
    UserValue getUserValue(){return userValue;}
    void setPassive(){userCmd = UserCommand::L2_B;}
    void setZero(){userValue.setZero();}
#ifdef COMPILE_WITH_REAL_ROBOT
    virtual void receiveHandle(UNITREE_LEGGED_SDK::LowState *lowState){};
#endif  // COMPILE_WITH_REAL_ROBOT
protected:
    virtual void* run(void *arg){return NULL;}
    UserCommand userCmd;
    UserValue userValue;
};

#endif  // CMDPANEL_H
