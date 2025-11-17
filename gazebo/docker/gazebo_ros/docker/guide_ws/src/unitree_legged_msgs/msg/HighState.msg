
uint8[2] head
uint8 levelFlag
uint8 frameReserve

uint32[2] SN
uint32[2] version
uint16 bandWidth
IMU imu
MotorState[20] motorState
BmsState bms
int16[4] footForce
int16[4] footForceEst
uint8 mode
float32 progress
uint8 gaitType
float32 footRaiseHeight
float32[3] position
float32 bodyHeight
float32[3] velocity
float32 yawSpeed
float32[4] rangeObstacle
Cartesian[4] footPosition2Body
Cartesian[4] footSpeed2Body
uint8[40] wirelessRemote
uint32 reserve

uint32 crc
