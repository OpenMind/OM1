/**********************************************************************
 Copyright (c) 2020-2023, Unitree Robotics.Co.Ltd. All rights reserved.
***********************************************************************/
#ifdef COMPILE_WITH_MOVE_BASE

#ifndef STATE_MOVE_BASE_H
#define STATE_MOVE_BASE_H

#include "FSM/State_Trotting.h"
#include "ros/ros.h"
#include <geometry_msgs/Twist.h>

class State_move_base : public State_Trotting{
public:
    State_move_base(CtrlComponents *ctrlComp);
    ~State_move_base(){}
    FSMStateName checkChange();
private:
    void getUserCmd();
    void initRecv();
    void twistCallback(const geometry_msgs::Twist& msg);
    ros::NodeHandle _nm;
    ros::Subscriber _cmdSub;
    double _vx, _vy;
    double _wz;
};

#endif  // STATE_MOVE_BASE_H

#endif  // COMPILE_WITH_MOVE_BASE
