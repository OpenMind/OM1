/************************************************************************
Copyright (c) 2018-2019, Unitree Robotics.Co.Ltd. All rights reserved.
Use of this source code is governed by the MPL-2.0 license, see LICENSE.
************************************************************************/

// #ifndef _UNITREE_JOINT_CONTROL_TOOL_H_
// #define _UNITREE_JOINT_CONTROL_TOOL_H_
#ifndef _LAIKAGO_CONTROL_TOOL_H_
#define _LAIKAGO_CONTROL_TOOL_H_

#include <stdio.h>
#include <stdint.h>
#include <algorithm>
#include <math.h>

#define posStopF (2.146E+9f)  // stop position control mode
#define velStopF (16000.0f)   // stop velocity control mode

typedef struct
{
    uint8_t mode;
    double pos;
    double posStiffness;
    double vel;
    double velStiffness;
    double torque;
} ServoCmd;

double clamp(double&, double, double);  // eg. clamp(1.5, -1, 1) = 1
double computeVel(double current_position, double last_position, double last_velocity, double duration);  // get current velocity
double computeTorque(double current_position, double current_velocity, ServoCmd&);  // get torque

#endif
