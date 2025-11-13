#ifndef MOTOR_MSG
#define MOTOR_MSG

#ifdef __cplusplus
extern "C"{
#endif

#include <stdint.h>
typedef int16_t q15_t;

#pragma pack(1)

// 发送用单个数据数据结构
typedef union{
        int32_t           L;
        uint8_t       u8[4];
       uint16_t      u16[2];
       uint32_t         u32;
          float           F;
}COMData32;

typedef struct {
	// 定义 数据包头
    unsigned char  start[2];     // 包头
	unsigned char  motorID;      // 电机ID  0,1,2,3 ...   0xBB 表示向所有电机广播（此时无返回）
	unsigned char  reserved;
}COMHead;

#pragma pack()

#pragma pack(1)

typedef struct {

	   uint8_t  fan_d;       // 关节上的散热风扇转速
	   uint8_t  Fmusic;      // 电机发声频率   /64*1000   15.625f 频率分度
	   uint8_t  Hmusic;      // 电机发声强度   推荐值4  声音强度 0.1 分度
	   uint8_t  reserved4;

	   uint8_t  FRGB[4];     // 足端LED

}LowHzMotorCmd;

typedef struct {  // 以 4个字节一组排列 ，不然编译器会凑整
	// 定义 数据
    uint8_t  mode;        // 关节模式选择
    uint8_t  ModifyBit;   // 电机控制参数修改位
    uint8_t  ReadBit;     // 电机控制参数发送位
    uint8_t  reserved;

    COMData32  Modify;     // 电机参数修改 的数据
    //实际给FOC的指令力矩为：
    //K_P*delta_Pos + K_W*delta_W + T
    q15_t     T;      // 期望关节的输出力矩（电机本身的力矩）x256, 7 + 8 描述
    q15_t     W;      // 期望关节速度 （电机本身的速度） x128,       8 + 7描述
    int32_t   Pos;      // 期望关节位置 x 16384/6.2832, 14位编码器（主控0点修正，电机关节还是以编码器0点为准）

    q15_t    K_P;      // 关节刚度系数 x2048  4+11 描述
    q15_t    K_W;      // 关节速度系数 x1024  5+10 描述

    uint8_t LowHzMotorCmdIndex;     // 电机低频率控制命令的索引, 0-7, 分别代表LowHzMotorCmd中的8个字节
    uint8_t LowHzMotorCmdByte;      // 电机低频率控制命令的字节

     COMData32  Res[1];    // 通讯 保留字节  用于实现别的一些通讯内容

}MasterComdV3;   // 加上数据包的包头 和CRC 34字节

typedef struct {
	// 定义 电机控制命令数据包
    COMHead head;
    MasterComdV3 Mdata;
    COMData32 CRCdata;
}MasterComdDataV3;//返回数据

// typedef struct {
// 	// 定义 总得485 数据包

//   MasterComdData M1;
// 	MasterComdData M2;
// 	MasterComdData M3;

// }DMA485TxDataV3;

#pragma pack()

#pragma pack(1)

typedef struct {  // 以 4个字节一组排列 ，不然编译器会凑整
    // 定义 数据
    uint8_t  mode;        // 当前关节模式
    uint8_t  ReadBit;     // 电机控制参数修改     是否成功位
    int8_t  Temp;        // 电机当前平均温度
    uint8_t  MError;      // 电机错误 标识

    COMData32  Read;     // 读取的当前 电机 的控制数据
    int16_t     T;      // 当前实际电机输出力矩       7 + 8 描述

    int16_t     W;      // 当前实际电机速度（高速）   8 + 7 描述
    float      LW;      // 当前实际电机速度（低速）

    int16_t     W2;      // 当前实际关节速度（高速）   8 + 7 描述
    float      LW2;      // 当前实际关节速度（低速）

    int16_t    Acc;           // 电机转子加速度       15+0 描述  惯量较小
    int16_t    OutAcc;        // 输出轴加速度         12+3 描述  惯量较大

    int32_t   Pos;      // 当前电机位置（主控0点修正，电机关节还是以编码器0点为准）
    int32_t   Pos2;     // 关节编码器位置(输出编码器)

    int16_t     gyro[3];  // 电机驱动板6轴传感器数据
    int16_t     acc[3];

    // 力传感器的数据
    int16_t     Fgyro[3];  //
    int16_t     Facc[3];
    int16_t     Fmag[3];
    uint8_t     Ftemp;     // 8位表示的温度  7位（-28~100度）  1位0.5度分辨率

    int16_t     Force16;   // 力传感器高16位数据
    int8_t      Force8;    // 力传感器低8位数据

    uint8_t     FError;    //  足端传感器错误标识

    int8_t      Res[1];    // 通讯 保留字节

}ServoComdV3;  // 加上数据包的包头 和CRC 78字节（4+70+4）

typedef struct {
    // 定义 电机控制命令数据包
    COMHead        head;
    ServoComdV3      Mdata;

    COMData32    CRCdata;

}ServoComdDataV3;	//发送数据

// typedef struct {
// 	// 定义 总的485 接受数据包

//   ServoComdDataV3 M[3];
//  // uint8_t  nullbyte1;

// }DMA485RxDataV3;


#pragma pack()

//  00 00 00 00 00
//  00 00 00 00 00
//  00 00 00 00 00
//  00 00 00
// 数据包默认初始化
// 主机发送的数据包
/*
                 Tx485Data[_FR][i].head.start[0] = 0xFE ;     Tx485Data[_FR][i].head.start[1] = 0xEE; // 数据包头
				 Tx485Data[_FR][i].Mdata.ModifyBit = 0xFF;    Tx485Data[_FR][i].Mdata.mode = 0;   // 默认不修改数据 和 电机的默认工作模式
				 Tx485Data[_FR][i].head.motorID = i;    0                                          // 目标电机标号
				 Tx485Data[_FR][i].Mdata.T = 0.0f;                           // 默认目标关节输出力矩                      motor1.Extra_Torque = motorRxData[1].Mdata.T*0.390625f;     // N.M  转化为 N.CM   IQ8描述
				 Tx485Data[_FR][i].Mdata.Pos = 0x7FE95C80;                   // 默认目标关节位置  不启用位置环          14位分辨率
				 Tx485Data[_FR][i].Mdata.W = 16000.0f;                       // 默认目标关节速度  不启用速度环          1+8+7描述     motor1.Target_Speed =  motorRxData[1].Mdata.W*0.0078125f;   // 单位 rad/s	       IQ7描述
				 Tx485Data[_FR][i].Mdata.K_P = (q15_t)(0.6f*(1<<11));        // 默认关节刚度系数   4+11 描述                     motor1.K_Pos = ((float)motorRxData[1].Mdata.K_P)/(1<<11);      // 描述刚度的通讯数据格式  4+11
				 Tx485Data[_FR][i].Mdata.K_W = (q15_t)(1.0f*(1<<10));        // 默认关节速度系数   5+10 描述                    motor1.K_Speed = ((float)motorRxData[1].Mdata.K_W)/(1<<10);    // 描述阻尼的通讯数据格式  5+10
*/

#ifdef __cplusplus
}
#endif

#endif
