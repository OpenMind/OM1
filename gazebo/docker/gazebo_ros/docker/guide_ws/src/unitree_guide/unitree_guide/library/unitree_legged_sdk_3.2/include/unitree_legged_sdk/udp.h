/************************************************************************
Copyright (c) 2020, Unitree Robotics.Co.Ltd. All rights reserved.
Use of this source code is governed by the MPL-2.0 license, see LICENSE.
************************************************************************/

#ifndef _UNITREE_LEGGED_UDP_H_
#define _UNITREE_LEGGED_UDP_H_

#include "comm.h"
#include "unitree_legged_sdk/quadruped.h"
#include <pthread.h>

namespace UNITREE_LEGGED_SDK
{

constexpr int UDP_CLIENT_PORT = 8080;                       // local port
constexpr int UDP_SERVER_PORT = 8007;                       // target port
constexpr char UDP_SERVER_IP_BASIC[] = "192.168.123.10";    // target IP address
constexpr char UDP_SERVER_IP_SPORT[] = "192.168.123.161";   // target IP address

// Notice: User defined data(like struct) should add crc(4Byte) at the end.
class UDP {
public:
    UDP(uint8_t level, HighLevelType highControl = HighLevelType::Basic);  // unitree dafault IP and Port
    UDP(uint16_t localPort, const char* targetIP, uint16_t targetPort, int sendLength, int recvLength);
    UDP(uint16_t localPort, uint16_t targetPort, int sendLength, int recvLength); // as server, client IP can change
    ~UDP();
    void InitCmdData(HighCmd& cmd);
    void InitCmdData(LowCmd& cmd);
    void switchLevel(int level);

    int SetSend(HighCmd&);
    int SetSend(LowCmd&);
    int SetSend(char* cmd);
    void GetRecv(HighState&);
    void GetRecv(LowState&);
    void GetRecv(char*);
    int Send();
    int Recv(); // directly save in buffer

    UDPState udpState;
    char*    targetIP;
    uint16_t targetPort;
    char*    localIP;
    uint16_t localPort;
private:
    void init(uint16_t localPort, const char* targetIP, uint16_t targetPort);

    uint8_t levelFlag = HIGHLEVEL;   // default: high level
    int sockFd;
    bool connected; // udp only works when connected
    int sendLength;
    int recvLength;
    char* recvTemp;
    char* recvBuf;
    char* sendBuf;
    int lose_recv;
    pthread_mutex_t sendMut;
    pthread_mutex_t recvMut;
};

}

#endif
