
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
uint32 tick
uint8[40] wirelessRemote
uint32 reserve
uint32 crc

# Old version Aliengo does not have:
Cartesian[4] eeForceRaw
Cartesian[4] eeForce          #it's a 1-DOF force infact, but we use 3-DOF here just for visualization
Cartesian position            # will delete
Cartesian velocity            # will delete
Cartesian velocity_w            # will delete
