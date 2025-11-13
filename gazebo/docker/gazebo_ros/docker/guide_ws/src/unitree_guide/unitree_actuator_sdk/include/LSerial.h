#ifndef LSERIAL
#define LSERIAL

#ifdef __cplusplus
extern "C"{
#endif

#include "motor_ctrl.h"

#if defined(__linux__)
    extern int open_set(char*);             //Input serial port name. (The Baud rate is fixed to 4800000), return the handle of serial port;【输入串口名。（波特率固定为4800000），返回串口句柄】
    extern int close_serial(int);           //Input serial port handle. Close the serial port【输入串口句柄，关闭串口】
    extern int broadcast(int, MOTOR_send*); //Input serial port handle and the pointer of message to be sent. Broadcast to all motors, and the motors do not have any respond【输入串口句柄和发送数据的指针，对电机广播，不接受返回】
    extern int send_recv(int, MOTOR_send*, MOTOR_recv*);    //Input serial port handle, the pointer of message to be sent and the pointer of received message.【输入串口句柄，发送数据的指针，接收数据的指针】
                                                            //Synchronous send and receive, send 34 bytes and receive 78 bytes【该函数为同步收发，每发一命令（34字节）必须接收一命令（78字节）】
                                                            //The return value is the status of send and receive. 0:send fail and receive fail, 1:send success and receive fail, 11:send success and receive success
                                                            //【返回值为发送接收状态，0：发送接收失败，1：发送成功接受失败，11：发送成功接收成功】
#elif defined(__WIN32__)
    #include <windows.h>
    extern HANDLE open_set(char*);             //Input serial port name. (The Baud rate is fixed to 4800000), return the handle of serial port;【输入串口名。（波特率固定为4800000），返回串口句柄】
    extern int close_serial(HANDLE);           //Input serial port handle. Close the serial port【输入串口句柄，关闭串口】
    extern int broadcast(HANDLE, MOTOR_send*); //Input serial port handle and the pointer of message to be sent. Broadcast to all motors, and the motors do not have any respond【输入串口句柄和发送数据的指针，对电机广播，不接受返回】
    extern int send_recv(HANDLE, MOTOR_send*, MOTOR_recv*);    //Input serial port handle, the pointer of message to be sent and the pointer of received message.【输入串口句柄，发送数据的指针，接收数据的指针】
                                                            //Synchronous send and receive, send 34 bytes and receive 78 bytes【该函数为同步收发，每发一命令（34字节）必须接收一命令（78字节）】
                                                            //The return value is the status of send and receive. 0:send fail and receive fail, 1:send success and receive fail, 11:send success and receive success
                                                            //【返回值为发送接收状态，0：发送接收失败，1：发送成功接受失败，11：发送成功接收成功】
#endif

#ifdef __cplusplus
}
#endif

#endif
