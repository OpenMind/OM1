/*****************************************************************
 Copyright (c) 2020, Unitree Robotics.Co.Ltd. All rights reserved.
******************************************************************/

#ifndef _UNITREE_LEGGED_GO1_H_
#define _UNITREE_LEGGED_GO1_H_

namespace UNITREE_LEGGED_SDK
{
    constexpr double go1_Hip_max   = 1.047;    // unit:radian ( = 60   degree)
    constexpr double go1_Hip_min   = -1.047;   // unit:radian ( = -60  degree)
    constexpr double go1_Thigh_max = 2.966;    // unit:radian ( = 170  degree)
    constexpr double go1_Thigh_min = -0.663;   // unit:radian ( = -38  degree)
    constexpr double go1_Calf_max  = -0.837;   // unit:radian ( = -48  degree)
    constexpr double go1_Calf_min  = -2.721;   // unit:radian ( = -156 degree)
}

#endif
