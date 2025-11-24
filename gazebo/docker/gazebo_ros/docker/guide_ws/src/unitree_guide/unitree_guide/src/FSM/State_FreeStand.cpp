/**********************************************************************
 Copyright (c) 2020-2023, Unitree Robotics.Co.Ltd. All rights reserved.
***********************************************************************/
#include "FSM/State_FreeStand.h"

State_FreeStand::State_FreeStand(CtrlComponents *ctrlComp)
             :FSMState(ctrlComp, FSMStateName::FREESTAND, "free stand"){
    _rowMax = 20 * M_PI / 180;
    _rowMin = -_rowMax;
    _pitchMax = 15 * M_PI / 180;
    _pitchMin = -_pitchMax;
    _yawMax = 20 * M_PI / 180;
    _yawMin = -_yawMax;
    _heightMax = 0.04;
    _heightMin = -_heightMax;
}

void State_FreeStand::enter(){
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

    for(int i=0; i<12; i++){
        _lowCmd->motorCmd[i].q = _lowState->motorState[i].q;
    }
    _initVecOX = _ctrlComp->robotModel->getX(*_lowState);
    _initVecXP = _ctrlComp->robotModel->getVecXP(*_lowState);

    _ctrlComp->setAllStance();
    _ctrlComp->ioInter->zeroCmdPanel();
}

void State_FreeStand::run(){
    Vec34 vecOP;
    _userValue = _lowState->userValue;

    vecOP = _calcOP( invNormalize(_userValue.lx, _rowMin, _rowMax),
                     invNormalize(_userValue.ly, _pitchMin, _pitchMax),
                    -invNormalize(_userValue.rx, _yawMin, _yawMax),
                     invNormalize(_userValue.ry, _heightMin, _heightMax) );
    _calcCmd(vecOP);
}

void State_FreeStand::exit(){
    _ctrlComp->ioInter->zeroCmdPanel();
}

FSMStateName State_FreeStand::checkChange(){
    if(_lowState->userCmd == UserCommand::L2_A){
        return FSMStateName::FIXEDSTAND;
    }
    else if(_lowState->userCmd == UserCommand::L2_B){
        return FSMStateName::PASSIVE;
    }
    else if(_lowState->userCmd == UserCommand::START){
        return FSMStateName::TROTTING;
    }
    else{
        return FSMStateName::FREESTAND;
    }
}

Vec34 State_FreeStand::_calcOP(float row, float pitch, float yaw, float height){
    Vec3 vecXO = -_initVecOX;
    vecXO(2) += height;

    RotMat rotM = rpyToRotMat(row, pitch, yaw);

    HomoMat Tsb = homoMatrix(vecXO, rotM);
    HomoMat Tbs = homoMatrixInverse(Tsb);

    Vec4 tempVec4;
    Vec34 vecOP;
    for(int i(0); i<4; ++i){
        tempVec4 = Tbs * homoVec(_initVecXP.col(i));
        vecOP.col(i) = noHomoVec(tempVec4);
    }

    return vecOP;
}

void State_FreeStand::_calcCmd(Vec34 vecOP){
    Vec12 q = _ctrlComp->robotModel->getQ(vecOP, FrameType::BODY);
    _lowCmd->setQ(q);
}
