/**********************************************************************
 Copyright (c) 2020-2023, Unitree Robotics.Co.Ltd. All rights reserved.
***********************************************************************/
#include "FSM/State_SwingTest.h"

State_SwingTest::State_SwingTest(CtrlComponents *ctrlComp)
                :FSMState(ctrlComp, FSMStateName::SWINGTEST, "swingTest"){
    _xMin = -0.15;
    _xMax =  0.10;
    _yMin = -0.15;
    _yMax =  0.15;
    _zMin = -0.05;
    _zMax =  0.20;
}

void State_SwingTest::enter(){
    for(int i=0; i<4; i++){
        if(_ctrlComp->ctrlPlatform == CtrlPlatform::GAZEBO){
            _lowCmd->setSimStanceGain(i);
        }
        else if(_ctrlComp->ctrlPlatform == CtrlPlatform::REALROBOT){
            _lowCmd->setRealStanceGain(i);
        }
        _lowCmd->setZeroDq(i);
        _lowCmd->setZeroTau(i);
    }
    _lowCmd->setSwingGain(0);

    _Kp = Vec3(20, 20, 50).asDiagonal();
    _Kd = Vec3( 5,  5, 20).asDiagonal();

    for(int i=0; i<12; i++){
        _lowCmd->motorCmd[i].q = _lowState->motorState[i].q;
    }

    _initFeetPos = _ctrlComp->robotModel->getFeet2BPositions(*_lowState, FrameType::HIP);
    _feetPos = _initFeetPos;
    _initPos = _initFeetPos.col(0);

    _ctrlComp->setAllSwing();
}

void State_SwingTest::run(){
    _userValue = _lowState->userValue;

    if(_userValue.ly > 0){
        _posGoal(0) = invNormalize(_userValue.ly, _initPos(0), _initPos(0)+_xMax, 0, 1);
    }else{
        _posGoal(0) = invNormalize(_userValue.ly, _initPos(0)+_xMin, _initPos(0), -1, 0);
    }

    if(_userValue.lx > 0){
        _posGoal(1) = invNormalize(_userValue.lx, _initPos(1, 0), _initPos(1)+_yMax, 0, 1);
    }else{
        _posGoal(1) = invNormalize(_userValue.lx, _initPos(1)+_yMin, _initPos(1), -1, 0);
    }

    if(_userValue.ry > 0){
        _posGoal(2) = invNormalize(_userValue.ry, _initPos(2), _initPos(2)+_zMax, 0, 1);
    }else{
        _posGoal(2) = invNormalize(_userValue.ry, _initPos(2)+_zMin, _initPos(2), -1, 0);
    }

    _positionCtrl();
    _torqueCtrl();
}

void State_SwingTest::exit(){
    _ctrlComp->ioInter->zeroCmdPanel();
}

FSMStateName State_SwingTest::checkChange(){
    if(_lowState->userCmd == UserCommand::L2_B){
        return FSMStateName::PASSIVE;
    }
    else if(_lowState->userCmd == UserCommand::L2_A){
        return FSMStateName::FIXEDSTAND;
    }
    else{
        return FSMStateName::SWINGTEST;
    }
}

void State_SwingTest::_positionCtrl(){
    _feetPos.col(0) = _posGoal;
    _targetPos = _ctrlComp->robotModel->getQ(_feetPos, FrameType::HIP);
    _lowCmd->setQ(_targetPos);
}

void State_SwingTest::_torqueCtrl(){
    Vec3 pos0 = _ctrlComp->robotModel->getFootPosition(*_lowState, 0, FrameType::HIP);
    Vec3 vel0 = _ctrlComp->robotModel->getFootVelocity(*_lowState, 0);

    Vec3 force0 = _Kp*(_posGoal - pos0) + _Kd*(-vel0);

    Vec12 torque;
    torque.setZero();
    Mat3 jaco0 = _ctrlComp->robotModel->getJaco(*_lowState, 0);

    torque.segment(0, 3) = jaco0.transpose() * force0;

    _lowCmd->setTau(torque);
}
