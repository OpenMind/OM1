#include "unitree_joint_control_tool.h"

float clamp(float &val, float min_val, float max_val)
{
    val = std::min(std::max(val, min_val), max_val);
}

double clamp(double &val, double min_val, double max_val)
{
    val = std::min(std::max(val, min_val), max_val);
}

double computeVel(double currentPos, double lastPos, double lastVel, double period)
{
    return lastVel*0.35f + 0.65f*(currentPos-lastPos)/period;
}

double computeTorque(double currentPos, double currentVel, ServoCmd &cmd)
{
    double targetPos, targetVel, targetTorque, posStiffness, velStiffness, calcTorque;
    targetPos = cmd.pos;
    targetVel = cmd.vel;
    targetTorque = cmd.torque;
    posStiffness = cmd.posStiffness;
    velStiffness = cmd.velStiffness;
    if(fabs(targetPos-posStopF) < 1e-6) posStiffness = 0;
    if(fabs(targetVel-velStopF) < 1e-6) velStiffness = 0;
    calcTorque = posStiffness*(targetPos-currentPos) + velStiffness*(targetVel-currentVel) + targetTorque;
    return calcTorque;
}
