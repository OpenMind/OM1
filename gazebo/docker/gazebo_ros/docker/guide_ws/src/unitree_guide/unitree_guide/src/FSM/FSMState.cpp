/**********************************************************************
 Copyright (c) 2020-2023, Unitree Robotics.Co.Ltd. All rights reserved.
***********************************************************************/
#include "FSM/FSMState.h"

FSMState::FSMState(CtrlComponents *ctrlComp, FSMStateName stateName, std::string stateNameString)
            :_ctrlComp(ctrlComp), _stateName(stateName), _stateNameString(stateNameString){
    _lowCmd = _ctrlComp->lowCmd;
    _lowState = _ctrlComp->lowState;
}
