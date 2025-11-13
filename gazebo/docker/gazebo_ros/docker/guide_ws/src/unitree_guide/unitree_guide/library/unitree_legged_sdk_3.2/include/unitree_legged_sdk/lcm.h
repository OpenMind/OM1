/************************************************************************
Copyright (c) 2020, Unitree Robotics.Co.Ltd. All rights reserved.
Use of this source code is governed by the MPL-2.0 license, see LICENSE.
************************************************************************/

#ifndef _UNITREE_LEGGED_LCM_H_
#define _UNITREE_LEGGED_LCM_H_

#include "comm.h"
#include <lcm/lcm-cpp.hpp>
#include <string.h>

namespace UNITREE_LEGGED_SDK
{

    constexpr char highCmdChannel[]   = "LCM_High_Cmd";
    constexpr char highStateChannel[] = "LCM_High_State";
    constexpr char lowCmdChannel[]    = "LCM_Low_Cmd";
    constexpr char lowStateChannel[]  = "LCM_Low_State";

    template<class T>
    class LCMHandler
    {
    public:
        LCMHandler(){
            pthread_mutex_init(&countMut, NULL);
            pthread_mutex_init(&recvMut, NULL);
        }

        void onMsg(const lcm::ReceiveBuffer* rbuf, const std::string& channel){
            isrunning = true;

            pthread_mutex_lock(&countMut);
            counter = 0;
            pthread_mutex_unlock(&countMut);

            T *msg = NULL;
            msg = (T *)(rbuf->data);
            pthread_mutex_lock(&recvMut);
            // sourceBuf = *msg;
            memcpy(&sourceBuf, msg, sizeof(T));
            pthread_mutex_unlock(&recvMut);
        }

        bool isrunning = false;
        T sourceBuf = {0};
        pthread_mutex_t countMut;
        pthread_mutex_t recvMut;
        int counter = 0;
    };

    class LCM {
	public:
        LCM(uint8_t level);
        ~LCM();
        void SubscribeCmd();
        void SubscribeState();         // remember to call this when change control level
        int Send(HighCmd&);            // lcm send cmd
        int Send(LowCmd&);             // lcm send cmd
        int Send(HighState&);          // lcm send state
        int Send(LowState&);           // lcm send state
		int Recv();                    // directly save in buffer
        void Get(HighCmd&);
        void Get(LowCmd&);
        void Get(HighState&);
        void Get(LowState&);

        LCMHandler<HighState>   highStateLCMHandler;
        LCMHandler<LowState>    lowStateLCMHandler;
        LCMHandler<HighCmd>     highCmdLCMHandler;
        LCMHandler<LowCmd>      lowCmdLCMHandler;
    private:
        uint8_t levelFlag = HIGHLEVEL;   // default: high level
        lcm::LCM lcm;
        lcm::Subscription* subLcm;
        int lcmFd;
        int LCM_PERIOD = 2000;     //default 2ms
	};

}

#endif
