/************************************************************************
Copyright (c) 2018-2019, Unitree Robotics.Co.Ltd. All rights reserved.
Use of this source code is governed by the MPL-2.0 license, see LICENSE.
************************************************************************/

// #include "unitree_legged_control/joint_controller.h"
#include "joint_controller.h"
#include <pluginlib/class_list_macros.h>

// #define rqtTune // use rqt or not

namespace unitree_legged_control
{

    UnitreeJointController::UnitreeJointController(){
        memset(&lastCmd, 0, sizeof(unitree_legged_msgs::MotorCmd));
        memset(&lastState, 0, sizeof(unitree_legged_msgs::MotorState));
        memset(&servoCmd, 0, sizeof(ServoCmd));
    }

    UnitreeJointController::~UnitreeJointController(){
        sub_ft.shutdown();
        sub_cmd.shutdown();
    }

    void UnitreeJointController::setTorqueCB(const geometry_msgs::WrenchStampedConstPtr& msg)
    {
        if(isHip) sensor_torque = msg->wrench.torque.x;
        else sensor_torque = msg->wrench.torque.y;
        // printf("sensor torque%f\n", sensor_torque);
    }

    void UnitreeJointController::setCommandCB(const unitree_legged_msgs::MotorCmdConstPtr& msg)
    {
        lastCmd.mode = msg->mode;
        lastCmd.q = msg->q;
        lastCmd.Kp = msg->Kp;
        lastCmd.dq = msg->dq;
        lastCmd.Kd = msg->Kd;
        lastCmd.tau = msg->tau;
        // the writeFromNonRT can be used in RT, if you have the guarantee that
        //  * no non-rt thread is calling the same function (we're not subscribing to ros callbacks)
        //  * there is only one single rt thread
        command.writeFromNonRT(lastCmd);
    }

    // Controller initialization in non-realtime
    bool UnitreeJointController::init(hardware_interface::EffortJointInterface *robot, ros::NodeHandle &n)
    {
        isHip = false;
        isThigh = false;
        isCalf = false;
        // rqtTune = false;
        sensor_torque = 0;
        name_space = n.getNamespace();
        if (!n.getParam("joint", joint_name)){
            ROS_ERROR("No joint given in namespace: '%s')", n.getNamespace().c_str());
            return false;
        }

        // load pid param from ymal only if rqt need
        // if(rqtTune) {
#ifdef rqtTune
            // Load PID Controller using gains set on parameter server
            if (!pid_controller_.init(ros::NodeHandle(n, "pid")))
                return false;
#endif
        // }

        urdf::Model urdf; // Get URDF info about joint
        if (!urdf.initParamWithNodeHandle("robot_description", n)){
            ROS_ERROR("Failed to parse urdf file");
            return false;
        }
        joint_urdf = urdf.getJoint(joint_name);
        if (!joint_urdf){
            ROS_ERROR("Could not find joint '%s' in urdf", joint_name.c_str());
            return false;
        }
        if(joint_name == "FR_hip_joint" || joint_name == "FL_hip_joint" || joint_name == "RR_hip_joint" || joint_name == "RL_hip_joint"){
            isHip = true;
        }
        if(joint_name == "FR_calf_joint" || joint_name == "FL_calf_joint" || joint_name == "RR_calf_joint" || joint_name == "RL_calf_joint"){
            isCalf = true;
        }
        joint = robot->getHandle(joint_name);

        // Start command subscriber
        sub_ft = n.subscribe(name_space + "/" +"joint_wrench", 1, &UnitreeJointController::setTorqueCB, this);
        sub_cmd = n.subscribe("command", 20, &UnitreeJointController::setCommandCB, this);

        // pub_state = n.advertise<unitree_legged_msgs::MotorState>(name_space + "/state", 20);
        // Start realtime state publisher
        controller_state_publisher_.reset(
            new realtime_tools::RealtimePublisher<unitree_legged_msgs::MotorState>(n, name_space + "/state", 1));

        return true;
    }

    void UnitreeJointController::setGains(const double &p, const double &i, const double &d, const double &i_max, const double &i_min, const bool &antiwindup)
    {
        pid_controller_.setGains(p,i,d,i_max,i_min,antiwindup);
    }

    void UnitreeJointController::getGains(double &p, double &i, double &d, double &i_max, double &i_min, bool &antiwindup)
    {
        pid_controller_.getGains(p,i,d,i_max,i_min,antiwindup);
    }

    void UnitreeJointController::getGains(double &p, double &i, double &d, double &i_max, double &i_min)
    {
        bool dummy;
        pid_controller_.getGains(p,i,d,i_max,i_min,dummy);
    }

    // Controller startup in realtime
    void UnitreeJointController::starting(const ros::Time& time)
    {
        // lastCmd.Kp = 0;
        // lastCmd.Kd = 0;
        double init_pos = joint.getPosition();
        lastCmd.q = init_pos;
        lastState.q = init_pos;
        lastCmd.dq = 0;
        lastState.dq = 0;
        lastCmd.tau = 0;
        lastState.tauEst = 0;
        command.initRT(lastCmd);

        pid_controller_.reset();
    }

    // Controller update loop in realtime
    void UnitreeJointController::update(const ros::Time& time, const ros::Duration& period)
    {
        double currentPos, currentVel, calcTorque;
        lastCmd = *(command.readFromRT());

        // set command data
        if(lastCmd.mode == PMSM) {
            servoCmd.pos = lastCmd.q;
            positionLimits(servoCmd.pos);
            servoCmd.posStiffness = lastCmd.Kp;
            if(fabs(lastCmd.q - PosStopF) < 0.00001){
                servoCmd.posStiffness = 0;
            }
            servoCmd.vel = lastCmd.dq;
            velocityLimits(servoCmd.vel);
            servoCmd.velStiffness = lastCmd.Kd;
            if(fabs(lastCmd.dq - VelStopF) < 0.00001){
                servoCmd.velStiffness = 0;
            }
            servoCmd.torque = lastCmd.tau;
            effortLimits(servoCmd.torque);
        }
        if(lastCmd.mode == BRAKE) {
            servoCmd.posStiffness = 0;
            servoCmd.vel = 0;
            servoCmd.velStiffness = 20;
            servoCmd.torque = 0;
            effortLimits(servoCmd.torque);
        }

        // } else {
        //     servoCmd.posStiffness = 0;
        //     servoCmd.velStiffness = 5;
        //     servoCmd.torque = 0;
        // }

        // rqt set P D gains
        // if(rqtTune) {
#ifdef rqtTune
            double i, i_max, i_min;
            getGains(servoCmd.posStiffness,i,servoCmd.velStiffness,i_max,i_min);
#endif
        // }

        currentPos = joint.getPosition();
        currentVel = computeVel(currentPos, (double)lastState.q, (double)lastState.dq, period.toSec());
        calcTorque = computeTorque(currentPos, currentVel, servoCmd);
        effortLimits(calcTorque);

        joint.setCommand(calcTorque);

        lastState.q = currentPos;
        lastState.dq = currentVel;
        // lastState.tauEst = calcTorque;
        // lastState.tauEst = sensor_torque;
        lastState.tauEst = joint.getEffort();

        // pub_state.publish(lastState);
        // publish state
        if (controller_state_publisher_ && controller_state_publisher_->trylock()) {
            controller_state_publisher_->msg_.q = lastState.q;
            controller_state_publisher_->msg_.dq = lastState.dq;
            controller_state_publisher_->msg_.tauEst = lastState.tauEst;
            controller_state_publisher_->unlockAndPublish();
        }

        // printf("sensor torque%f\n", sensor_torque);

        // if(joint_name == "wrist1_joint") printf("wrist1 setp:%f  getp:%f t:%f\n", servoCmd.pos, currentPos, calcTorque);
    }

    // Controller stopping in realtime
    void UnitreeJointController::stopping(){}

    void UnitreeJointController::positionLimits(double &position)
    {
        if (joint_urdf->type == urdf::Joint::REVOLUTE || joint_urdf->type == urdf::Joint::PRISMATIC)
            clamp(position, joint_urdf->limits->lower, joint_urdf->limits->upper);
    }

    void UnitreeJointController::velocityLimits(double &velocity)
    {
        if (joint_urdf->type == urdf::Joint::REVOLUTE || joint_urdf->type == urdf::Joint::PRISMATIC)
            clamp(velocity, -joint_urdf->limits->velocity, joint_urdf->limits->velocity);
    }

    void UnitreeJointController::effortLimits(double &effort)
    {
        if (joint_urdf->type == urdf::Joint::REVOLUTE || joint_urdf->type == urdf::Joint::PRISMATIC)
            clamp(effort, -joint_urdf->limits->effort, joint_urdf->limits->effort);
    }

} // namespace

// Register controller to pluginlib
PLUGINLIB_EXPORT_CLASS(unitree_legged_control::UnitreeJointController, controller_interface::ControllerBase);
