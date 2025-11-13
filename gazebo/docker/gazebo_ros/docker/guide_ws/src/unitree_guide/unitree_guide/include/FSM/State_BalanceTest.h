/**********************************************************************
 Copyright (c) 2020-2023, Unitree Robotics.Co.Ltd. All rights reserved.
***********************************************************************/
#ifndef BALANCETEST_H
#define BALANCETEST_H

#include "FSM/FSMState.h"

class State_BalanceTest : public FSMState{
public:
    State_BalanceTest(CtrlComponents *ctrlComp);
    ~State_BalanceTest(){}
    void enter();
    void run();
    void exit();
    FSMStateName checkChange();
private:
    void calcTau();

    Estimator *_est;
    QuadrupedRobot *_robModel;
    BalanceCtrl *_balCtrl;

    VecInt4 *_contact;

    RotMat _Rd, _RdInit;
    Vec3 _pcd, _pcdInit;
    double _kpw;
    Mat3 _Kpp, _Kdp, _Kdw;
    Vec3 _ddPcd, _dWbd;

    Vec12 _q, _tau;
    Vec3 _posBody, _velBody;
    RotMat _B2G_RotMat, _G2B_RotMat;
    Vec34 _posFeet2BGlobal;
    Vec34 _forceFeetGlobal, _forceFeetBody;

    float _xMax, _xMin;
    float _yMax, _yMin;
    float _zMax, _zMin;
    float _yawMax, _yawMin;
};

#endif  // BALANCETEST_H
