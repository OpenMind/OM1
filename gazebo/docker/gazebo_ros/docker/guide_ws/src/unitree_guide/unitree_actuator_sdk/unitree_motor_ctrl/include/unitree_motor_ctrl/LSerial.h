#ifndef LSERIAL
#define LSERIAL

#ifdef __cplusplus
extern "C"{
#endif

#include "motor_ctrl.h"

#if defined(__linux__)
    extern int open_set(char*);             //串口名。（波特率固定为4800000），返回串口句柄
    extern int close_serial(int);           //关闭串口
    extern int broadcast(int, MOTOR_send*); //对电机广播，不接受返回
    extern int send_recv(int, MOTOR_send*, MOTOR_recv*);     //该函数为同步收发，每发一命令（34字节）必须接收一命令（78字节）
                                                                    //返回值为发送接收状态，0：发送接收失败，1：发送成功接受失败，10：发送失败接收成功（不可能），11：发送成功接收成功
#elif defined(__WIN32__)
    #include <windows.h>
    extern HANDLE open_set(char*);             //串口名。（波特率固定为4800000），返回串口句柄
    extern int close_serial(HANDLE);
    extern int send_recv(HANDLE, MOTOR_send*, MOTOR_recv*);     //该函数为同步收发，每发一命令（34字节）必须接收一命令（78字节）
                                                                    //返回值为发送接收状态，0：发送接收失败，1：发送成功接受失败，10：发送失败接收成功（不可能），11：发送成功接收成功
#endif

#ifdef __cplusplus
}
#endif

#endif
