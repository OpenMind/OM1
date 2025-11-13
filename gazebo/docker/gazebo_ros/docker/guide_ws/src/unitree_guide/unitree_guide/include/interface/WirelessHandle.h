/**********************************************************************
 Copyright (c) 2020-2023, Unitree Robotics.Co.Ltd. All rights reserved.
***********************************************************************/
#ifndef WIRELESSHANDLE_H
#define WIRELESSHANDLE_H

#include "message/unitree_joystick.h"
#include "interface/CmdPanel.h"
#include "unitree_legged_sdk/comm.h"

class WirelessHandle : public CmdPanel{
public:
    WirelessHandle();
    ~WirelessHandle(){}
    void receiveHandle(UNITREE_LEGGED_SDK::LowState *lowState);
private:
    xRockerBtnDataStruct _keyData;
};

#endif  // WIRELESSHANDLE_H
