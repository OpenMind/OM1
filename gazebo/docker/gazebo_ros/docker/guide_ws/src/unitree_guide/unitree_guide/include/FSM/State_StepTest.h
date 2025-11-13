/**********************************************************************
 Copyright (c) 2020-2023, Unitree Robotics.Co.Ltd. All rights reserved.
***********************************************************************/
#ifndef STEPTEST_H
#define STEPTEST_H

#include "FSM/FSMState.h"

class State_StepTest : public FSMState{
public:
    State_StepTest(CtrlComponents *ctrlComp);
    ~State_StepTest(){}
    void enter();
    void run();
    void exit();
    FSMStateName checkChange();
private:
    void calcTau();

    float _gaitHeight;

    Estimator *_est;
    QuadrupedRobot *_robModel;
    BalanceCtrl *_balCtrl;

    VecInt4 *_contact;
    Vec4 *_phase;

    RotMat _Rd;
    Vec3 _pcd;
    Mat3 _Kpp, _Kpw, _Kdp, _Kdw;
    Mat3 _KpSwing, _KdSwing;
    Vec3 _ddPcd, _dWbd;

    Vec12 _q, _tau;
    Vec3 _posBody, _velBody;
    RotMat _B2G_RotMat, _G2B_RotMat;
    Vec34 _posFeet2BGlobal;
    Vec34 _posFeetGlobalInit, _posFeetGlobalGoal, _velFeetGlobalGoal;
    Vec34 _posFeetGlobal, _velFeetGlobal;
    Vec34 _forceFeetGlobal, _forceFeetBody;
};

#endif  // STEPTEST_H
