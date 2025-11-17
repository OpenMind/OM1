/**********************************************************************
 Copyright (c) 2020-2023, Unitree Robotics.Co.Ltd. All rights reserved.
***********************************************************************/
#ifndef CONTROLFRAME_H
#define CONTROLFRAME_H

#include "FSM/FSM.h"
#include "control/CtrlComponents.h"

class ControlFrame{
public:
	ControlFrame(CtrlComponents *ctrlComp);
	~ControlFrame(){
		delete _FSMController;
	}
	void run();
private:
	FSM* _FSMController;
	CtrlComponents *_ctrlComp;
};

#endif  //CONTROLFRAME_H
