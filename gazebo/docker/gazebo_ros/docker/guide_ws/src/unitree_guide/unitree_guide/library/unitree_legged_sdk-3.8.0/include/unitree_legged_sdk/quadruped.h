/*****************************************************************
 Copyright (c) 2020, Unitree Robotics.Co.Ltd. All rights reserved.
******************************************************************/

#ifndef _UNITREE_LEGGED_QUADRUPED_H_
#define _UNITREE_LEGGED_QUADRUPED_H_

#include <string>

using namespace std;

namespace UNITREE_LEGGED_SDK
{

enum class LeggedType {
	Aliengo,
	A1,
	Go1,
	B1
};


string VersionSDK();
int InitEnvironment();      // memory lock

// definition of each leg and joint
constexpr int FR_ = 0;       // leg index
constexpr int FL_ = 1;
constexpr int RR_ = 2;
constexpr int RL_ = 3;

constexpr int FR_0 = 0;      // joint index
constexpr int FR_1 = 1;
constexpr int FR_2 = 2;

constexpr int FL_0 = 3;
constexpr int FL_1 = 4;
constexpr int FL_2 = 5;

constexpr int RR_0 = 6;
constexpr int RR_1 = 7;
constexpr int RR_2 = 8;

constexpr int RL_0 = 9;
constexpr int RL_1 = 10;
constexpr int RL_2 = 11;

}

#endif
