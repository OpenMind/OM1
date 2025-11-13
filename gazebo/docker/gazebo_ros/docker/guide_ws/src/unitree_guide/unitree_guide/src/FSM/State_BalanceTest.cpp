/**********************************************************************
 Copyright (c) 2020-2023, Unitree Robotics.Co.Ltd. All rights reserved.
***********************************************************************/
#include "FSM/State_BalanceTest.h"

State_BalanceTest::State_BalanceTest(CtrlComponents *ctrlComp)
                  :FSMState(ctrlComp, FSMStateName::BALANCETEST, "balanceTest"),
                  _est(ctrlComp->estimator), _robModel(ctrlComp->robotModel),
                  _balCtrl(ctrlComp->balCtrl), _contact(ctrlComp->contact){

    _xMax = 0.05;
    _xMin = -_xMax;
    _yMax = 0.05;
    _yMin = -_yMax;
    _zMax = 0.04;
    _zMin = -_zMax;
    _yawMax = 20 * M_PI / 180;
    _yawMin = -_yawMax;

    _Kpp = Vec3(150, 150, 150).asDiagonal();
    _Kdp = Vec3(25, 25, 25).asDiagonal();

    _kpw = 200;
    _Kdw = Vec3(30, 30, 30).asDiagonal();
}

void State_BalanceTest::enter(){
    _pcdInit = _est->getPosition();
    _pcd = _pcdInit;
    _RdInit = _lowState->getRotMat();

    _ctrlComp->setAllStance();
    _ctrlComp->ioInter->zeroCmdPanel();
}

void State_BalanceTest::run(){
    _userValue = _lowState->userValue;

    _pcd(0) = _pcdInit(0) + invNormalize(_userValue.ly, _xMin, _xMax);
    _pcd(1) = _pcdInit(1) - invNormalize(_userValue.lx, _yMin, _yMax);
    _pcd(2) = _pcdInit(2) + invNormalize(_userValue.ry, _zMin, _zMax);

    float yaw = invNormalize(_userValue.rx, _yawMin, _yawMax);
    _Rd = rpyToRotMat(0, 0, yaw)*_RdInit;

    _posBody = _est->getPosition();
    _velBody = _est->getVelocity();

    _B2G_RotMat = _lowState->getRotMat();
    _G2B_RotMat = _B2G_RotMat.transpose();

    calcTau();

    _lowCmd->setStableGain();
    _lowCmd->setTau(_tau);
    _lowCmd->setQ(_q);
}

void State_BalanceTest::exit(){
    _ctrlComp->ioInter->zeroCmdPanel();
}

FSMStateName State_BalanceTest::checkChange(){
    if(_lowState->userCmd == UserCommand::L2_B){
        return FSMStateName::PASSIVE;
    }
    else if(_lowState->userCmd == UserCommand::L2_A){
        return FSMStateName::FIXEDSTAND;
    }
    else{
        return FSMStateName::BALANCETEST;
    }
}

void State_BalanceTest::calcTau(){

    _ddPcd = _Kpp*(_pcd - _posBody) + _Kdp * (Vec3(0, 0, 0) - _velBody);
    _dWbd  = _kpw*rotMatToExp(_Rd*_G2B_RotMat) + _Kdw * (Vec3(0, 0, 0) - _lowState->getGyroGlobal());

    _posFeet2BGlobal = _est->getPosFeet2BGlobal();

    _forceFeetGlobal = - _balCtrl->calF(_ddPcd, _dWbd, _B2G_RotMat, _posFeet2BGlobal, *_contact);
    _forceFeetBody = _G2B_RotMat * _forceFeetGlobal;

    _q = vec34ToVec12(_lowState->getQ());
    _tau = _robModel->getTau(_q, _forceFeetBody);
}
