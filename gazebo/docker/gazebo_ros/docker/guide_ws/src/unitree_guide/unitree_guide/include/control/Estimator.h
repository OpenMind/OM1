/**********************************************************************
 Copyright (c) 2020-2023, Unitree Robotics.Co.Ltd. All rights reserved.
***********************************************************************/
#ifndef ESTIMATOR_H
#define ESTIMATOR_H

#include <vector>
#include "common/unitreeRobot.h"
#include "common/LowPassFilter.h"
#include "Gait/WaveGenerator.h"
#include "message/LowlevelState.h"
#include "string"

#ifdef COMPILE_DEBUG
    #include "common/PyPlot.h"
#endif  // COMPILE_DEBUG

#ifdef COMPILE_WITH_MOVE_BASE
    #include <ros/ros.h>
    #include <ros/time.h>
    #include <geometry_msgs/TransformStamped.h>
    #include <tf/transform_broadcaster.h>
    #include <nav_msgs/Odometry.h>
    #include <geometry_msgs/Twist.h>
    #include <boost/array.hpp>
#endif  // COMPILE_WITH_MOVE_BASE

class Estimator{
public:
    Estimator(QuadrupedRobot *robotModel, LowlevelState* lowState, VecInt4 *contact, Vec4 *phase, double dt);
    Estimator(QuadrupedRobot *robotModel, LowlevelState* lowState, VecInt4 *contact, Vec4 *phase, double dt, Vec18 Qdig, std::string testName);
    ~Estimator();
    Vec3  getPosition();
    Vec3  getVelocity();
    Vec3  getFootPos(int i);
    Vec34 getFeetPos();
    Vec34 getFeetVel();
    Vec34 getPosFeet2BGlobal();
    void run();

#ifdef COMPILE_DEBUG
    void setPyPlot(PyPlot *plot){_testPlot = plot;}
#endif  // COMPILE_DEBUG

private:
    void _initSystem();
    // Linear System
    Eigen::Matrix<double, 18, 1>  _xhat;            // The state of estimator, position(3)+velocity(3)+feet position(3x4)
    Vec3 _u;                                        // The input of estimator
    Eigen::Matrix<double, 28,  1> _y;               // The measurement value of output y
    Eigen::Matrix<double, 28,  1> _yhat;            // The prediction of output y
    Eigen::Matrix<double, 18, 18> _A;               // The transtion matrix of estimator
    Eigen::Matrix<double, 18, 3>  _B;               // The input matrix
    Eigen::Matrix<double, 28, 18> _C;               // The output matrix
    // Covariance Matrix
    Eigen::Matrix<double, 18, 18> _P;               // Prediction covariance
    Eigen::Matrix<double, 18, 18> _Ppriori;         // Priori prediction covariance
    Eigen::Matrix<double, 18, 18> _Q;               // Dynamic simulation covariance
    Eigen::Matrix<double, 28, 28> _R;               // Measurement covariance
    Eigen::Matrix<double, 18, 18> _QInit;           // Initial value of Dynamic simulation covariance
    Eigen::Matrix<double, 28, 28> _RInit;           // Initial value of Measurement covariance
    Vec18 _Qdig;                                    // adjustable process noise covariance
    Mat3 _Cu;                                       // The covariance of system input u
    // Output Measurement
    Eigen::Matrix<double, 12, 1>  _feetPos2Body;    // The feet positions to body, in the global coordinate
    Eigen::Matrix<double, 12, 1>  _feetVel2Body;    // The feet velocity to body, in the global coordinate
    Eigen::Matrix<double,  4, 1>  _feetH;           // The Height of each foot, in the global coordinate
    Eigen::Matrix<double, 28, 28> _S;               // _S = C*P*C.T + R
    Eigen::PartialPivLU<Eigen::Matrix<double, 28, 28>> _Slu;    // _S.lu()
    Eigen::Matrix<double, 28,  1> _Sy;              // _Sy = _S.inv() * (y - yhat)
    Eigen::Matrix<double, 28, 18> _Sc;              // _Sc = _S.inv() * C
    Eigen::Matrix<double, 28, 28> _SR;              // _SR = _S.inv() * R
    Eigen::Matrix<double, 28, 18> _STC;             // _STC = (_S.transpose()).inv() * C
    Eigen::Matrix<double, 18, 18> _IKC;             // _IKC = I - KC

    RotMat _rotMatB2G;                              // Rotate Matrix: from body to global
    Vec3 _g;
    Vec34 _feetPosGlobalKine, _feetVelGlobalKine;

    LowlevelState* _lowState;
    QuadrupedRobot *_robModel;
    Vec4 *_phase;
    VecInt4 *_contact;
    double _dt;
    double _trust;
    double _largeVariance;

    // Low pass filters
    LPFilter *_vxFilter, *_vyFilter, *_vzFilter;

    // Tuning
    AvgCov *_RCheck;
    AvgCov *_uCheck;
    std::string _estName;

#ifdef COMPILE_DEBUG
    PyPlot *_testPlot;
#endif  // COMPILE_DEBUG
#ifdef COMPILE_WITH_MOVE_BASE
    ros::NodeHandle _nh;
    ros::Publisher _pub;
    tf::TransformBroadcaster _odomBroadcaster;
    ros::Time _currentTime;
    geometry_msgs::TransformStamped _odomTF;
    nav_msgs::Odometry _odomMsg;
    int _count = 0;
    double _pubFreq = 10;

    Vec3 _velBody, _wBody;
    boost::array<double, 36> _odom_pose_covariance = {1e-9, 0, 0, 0, 0, 0,
                                        0, 1e-3, 1e-9, 0, 0, 0,
                                        0, 0, 1e6, 0, 0, 0,
                                        0, 0, 0, 1e6, 0, 0,
                                        0, 0, 0, 0, 1e6, 0,
                                        0, 0, 0, 0, 0, 1e-9};
    boost::array<double, 36> _odom_twist_covariance = {1e-9, 0, 0, 0, 0, 0,
                                        0, 1e-3, 1e-9, 0, 0, 0,
                                        0, 0, 1e6, 0, 0, 0,
                                        0, 0, 0, 1e6, 0, 0,
                                        0, 0, 0, 0, 1e6, 0,
                                        0, 0, 0, 0, 0, 1e-9};
#endif  // COMPILE_WITH_MOVE_BASE

};

#endif  // ESTIMATOR_H
