/**********************************************************************
 Copyright (c) 2020-2023, Unitree Robotics.Co.Ltd. All rights reserved.
***********************************************************************/
#ifndef STATE_SWINGTEST_H
#define STATE_SWINGTEST_H

#include "FSM/FSMState.h"
#include "Gait/GaitGenerator.h"

class State_SwingTest : public FSMState{
public:
    State_SwingTest(CtrlComponents *ctrlComp);
    ~State_SwingTest(){};
    void enter();
    void run();
    void exit();
    FSMStateName checkChange();
private:
    void _positionCtrl();
    void _torqueCtrl();

    Vec34 _initFeetPos, _feetPos;
    Vec3  _initPos, _posGoal;
    Vec12 _targetPos;
    float _xMin, _xMax;
    float _yMin, _yMax;
    float _zMin, _zMax;
    Mat3 _Kp, _Kd;
};

#endif  // STATE_SWINGTEST_H
