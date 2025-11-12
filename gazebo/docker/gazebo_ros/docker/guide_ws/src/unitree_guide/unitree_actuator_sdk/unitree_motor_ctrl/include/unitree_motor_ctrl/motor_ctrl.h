#ifndef MOTOR_CTRL
#define MOTOR_CTRL

#ifdef __cplusplus
extern "C"{
#endif

#include "motor_msg.h"  //电机通信协议
#include <stdint.h>

typedef struct {
	// 定义 发送格式化数据
    MasterComdDataV3  motor_send_data;  //电机控制数据结构体，详见motor_msg.h
	int hex_len;                    //发送的16进制命令数组长度, 34
    long long send_time;            //The time that message was sent发送该命令的时间, 微秒(us)
    // 待发送的各项数据
    unsigned short id;              //电机ID，0代表全部电机
    unsigned short mode;            //0:空闲, 5:开环转动, 10:闭环FOC控制
    //实际给FOC的指令力矩为：
    //K_P*delta_Pos + K_W*delta_W + T
    float T;                        //期望关节的输出力矩（电机本身的力矩）（Nm）
    float W;                        //期望关节速度（电机本身的速度）(rad/s)
    float Pos;                      //期望关节位置（rad）
    float K_P;                      //关节刚度系数
    float K_W;                      //关节速度系数
}MOTOR_send;

typedef struct
{
    // 定义 接收数据
    ServoComdDataV3 motor_recv_data;     //电机接收数据结构体，详见motor_msg.h
    int hex_len;                    //接收的16进制命令数组长度, 78
    long long resv_time;            //接收该命令的时间, 微秒(us)
    int correct;                   //接收数据是否完整（1完整，0不完整）
    //解读得出的电机数据
    unsigned char motor_id;         //电机ID
    unsigned char mode;             //0:空闲, 5:开环转动, 10:闭环FOC控制
    int Temp;                       //温度
    unsigned char MError;           //错误码

    float T;                        // 当前实际电机输出力矩
    float W;                        // 当前实际电机速度（高速）
    float LW;                       // 当前实际电机速度（低速）
    int Acc;                      // 电机转子加速度
    float Pos;                      // 当前电机位置（主控0点修正，电机关节还是以编码器0点为准）

    float gyro[3];                  // 电机驱动板6轴传感器数据
    float acc[3];

}MOTOR_recv;

extern long long getSystemTime();       //获取当前系统时间（微秒us）
extern int modify_data(MOTOR_send*);    //将数据处理为stm32需求的格式
extern int extract_data(MOTOR_recv*);   //将接收到的数据解读
uint32_t crc32_core(uint32_t*, uint32_t);    //待校验数组指针，数组长度（余数舍去）

#ifdef __cplusplus
}
#endif

#endif
