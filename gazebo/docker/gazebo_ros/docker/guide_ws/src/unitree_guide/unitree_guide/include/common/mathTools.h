/**********************************************************************
 Copyright (c) 2020-2023, Unitree Robotics.Co.Ltd. All rights reserved.
***********************************************************************/
#ifndef MATHTOOLS_H
#define MATHTOOLS_H

#include <stdio.h>
#include <iostream>
#include "common/mathTypes.h"

template<typename T1, typename T2>
inline T1 max(const T1 a, const T2 b){
	return (a > b ? a : b);
}

template<typename T1, typename T2>
inline T1 min(const T1 a, const T2 b){
	return (a < b ? a : b);
}

template<typename T>
inline T saturation(const T a, Vec2 limits){
    T lowLim, highLim;
    if(limits(0) > limits(1)){
        lowLim = limits(1);
        highLim= limits(0);
    }else{
        lowLim = limits(0);
        highLim= limits(1);
    }

    if(a < lowLim){
        return lowLim;
    }
    else if(a > highLim){
        return highLim;
    }
    else{
        return a;
    }
}

template<typename T0, typename T1>
inline T0 killZeroOffset(T0 a, const T1 limit){
    if((a > -limit) && (a < limit)){
        a = 0;
    }
    return a;
}

template<typename T0, typename T1, typename T2>
inline T1 invNormalize(const T0 value, const T1 min, const T2 max, const double minLim = -1, const double maxLim = 1){
	return (value-minLim)*(max-min)/(maxLim-minLim) + min;
}

template<typename T>
inline T windowFunc(const T x, const T windowRatio, const T xRange=1.0, const T yRange=1.0){
    if((x < 0)||(x > xRange)){
        std::cout << "[ERROR][windowFunc] The x=" << x << ", which should between [0, xRange]" << std::endl;
    }
    if((windowRatio <= 0)||(windowRatio >= 0.5)){
        std::cout << "[ERROR][windowFunc] The windowRatio=" << windowRatio << ", which should between [0, 0.5]" << std::endl;
    }

    if(x/xRange < windowRatio){
        return x * yRange / (xRange * windowRatio);
    }
    else if(x/xRange > 1 - windowRatio){
        return yRange * (xRange - x)/(xRange * windowRatio);
    }
    else{
        return yRange;
    }
}

template<typename T1, typename T2>
inline void updateAverage(T1 &exp, T2 newValue, double n){
    if(exp.rows()!=newValue.rows()){
        std::cout << "The size of updateAverage is error" << std::endl;
        exit(-1);
    }
    if(fabs(n - 1) < 0.001){
        exp = newValue;
    }else{
        exp = exp + (newValue - exp)/n;
    }
}

template<typename T1, typename T2, typename T3>
inline void updateCovariance(T1 &cov, T2 expPast, T3 newValue, double n){
    if( (cov.rows()!=cov.cols()) || (cov.rows() != expPast.rows()) || (expPast.rows()!=newValue.rows())){
        std::cout << "The size of updateCovariance is error" << std::endl;
        exit(-1);
    }
    if(fabs(n - 1) < 0.1){
        cov.setZero();
    }else{
        cov = cov*(n-1)/n + (newValue-expPast)*(newValue-expPast).transpose()*(n-1)/(n*n);
    }
}

template<typename T1, typename T2, typename T3>
inline void updateAvgCov(T1 &cov, T2 &exp, T3 newValue, double n){
    // The order matters!!! covariance first!!!
    updateCovariance(cov, exp, newValue, n);
    updateAverage(exp, newValue, n);
}

inline RotMat rotx(const double &theta) {
    double s = std::sin(theta);
    double c = std::cos(theta);

    RotMat R;
    R << 1, 0, 0, 0, c, -s, 0, s, c;
    return R;
}

inline RotMat roty(const double &theta) {
    double s = std::sin(theta);
    double c = std::cos(theta);

    RotMat R;
    R << c, 0, s, 0, 1, 0, -s, 0, c;
    return R;
}

inline RotMat rotz(const double &theta) {
    double s = std::sin(theta);
    double c = std::cos(theta);

    RotMat R;
    R << c, -s, 0, s, c, 0, 0, 0, 1;
    return R;
}

inline Mat2 skew(const double& w){
    Mat2 mat; mat.setZero();
    mat(0, 1) = -w;
    mat(1, 0) =  w;
    return mat;
}

inline Mat3 skew(const Vec3& v) {
    Mat3 m;
    m << 0, -v(2), v(1),
            v(2), 0, -v(0),
            -v(1), v(0), 0;
    return m;
}

inline RotMat rpyToRotMat(const double& row, const double& pitch, const double& yaw) {
    RotMat m = rotz(yaw) * roty(pitch) * rotx(row);
    return m;
}

inline Vec3 rotMatToRPY(const Mat3& R) {
    Vec3 rpy;
    rpy(0) = atan2(R(2,1),R(2,2));
    rpy(1) = asin(-R(2,0));
    rpy(2) = atan2(R(1,0),R(0,0));
    return rpy;
}

inline RotMat quatToRotMat(const Quat& q) {
    double e0 = q(0);
    double e1 = q(1);
    double e2 = q(2);
    double e3 = q(3);

    RotMat R;
    R << 1 - 2 * (e2 * e2 + e3 * e3), 2 * (e1 * e2 - e0 * e3),
            2 * (e1 * e3 + e0 * e2), 2 * (e1 * e2 + e0 * e3),
            1 - 2 * (e1 * e1 + e3 * e3), 2 * (e2 * e3 - e0 * e1),
            2 * (e1 * e3 - e0 * e2), 2 * (e2 * e3 + e0 * e1),
            1 - 2 * (e1 * e1 + e2 * e2);
    return R;
}

inline Vec3 rotMatToExp(const RotMat& rm){
    double cosValue = rm.trace()/2.0-1/2.0;
    if(cosValue > 1.0f){
        cosValue = 1.0f;
    }else if(cosValue < -1.0f){
        cosValue = -1.0f;
    }

    double angle = acos(cosValue);
    Vec3 exp;
    if (fabs(angle) < 1e-5){
        exp=Vec3(0,0,0);
    }
    else if (fabs(angle - M_PI) < 1e-5){
        exp = angle * Vec3(rm(0,0)+1, rm(0,1), rm(0,2)) / sqrt(2*(1+rm(0, 0)));
    }
    else{
        exp=angle/(2.0f*sin(angle))*Vec3(rm(2,1)-rm(1,2),rm(0,2)-rm(2,0),rm(1,0)-rm(0,1));
    }
    return exp;
}

inline HomoMat homoMatrix(Vec3 p, RotMat m){
    HomoMat homoM;
    homoM.setZero();
    homoM.topLeftCorner(3, 3) = m;
    homoM.topRightCorner(3, 1) = p;
    homoM(3, 3) = 1;
    return homoM;
}

inline HomoMat homoMatrix(Vec3 p, Quat q){
    HomoMat homoM;
    homoM.setZero();
    homoM.topLeftCorner(3, 3) = quatToRotMat(q);
    homoM.topRightCorner(3, 1) = p;
    homoM(3, 3) = 1;
    return homoM;
}

inline HomoMat homoMatrixInverse(HomoMat homoM){
    HomoMat homoInv;
    homoInv.setZero();
    homoInv.topLeftCorner(3, 3) = homoM.topLeftCorner(3, 3).transpose();
    homoInv.topRightCorner(3, 1) = -homoM.topLeftCorner(3, 3).transpose() * homoM.topRightCorner(3, 1);
    homoInv(3, 3) = 1;
    return homoInv;
}

//  add 1 at the end of Vec3
inline Vec4 homoVec(Vec3 v3){
    Vec4 v4;
    v4.block(0, 0, 3, 1) = v3;
    v4(3) = 1;
    return v4;
}

//  remove 1 at the end of Vec4
inline Vec3 noHomoVec(Vec4 v4){
    Vec3 v3;
    v3 = v4.block(0, 0, 3, 1);
    return v3;
}

// Calculate average value and covariance
class AvgCov{
public:
    AvgCov(unsigned int size, std::string name, bool avgOnly=false, unsigned int showPeriod=1000, unsigned int waitCount=5000, double zoomFactor=10000)
            :_size(size), _showPeriod(showPeriod), _waitCount(waitCount), _zoomFactor(zoomFactor), _valueName(name), _avgOnly(avgOnly) {
        _exp.resize(size);
        _cov.resize(size, size);
        _defaultWeight.resize(size, size);
        _defaultWeight.setIdentity();
        _measureCount = 0;
    }
    void measure(VecX newValue){
        ++_measureCount;

        if(_measureCount > _waitCount){
            updateAvgCov(_cov, _exp, newValue, _measureCount-_waitCount);
            if(_measureCount % _showPeriod == 0){
                std::cout << "******" << _valueName << " measured count: " << _measureCount-_waitCount << "******" << std::endl;
                std::cout << _zoomFactor << " Times Average of " << _valueName << std::endl << (_zoomFactor*_exp).transpose() << std::endl;
                if(!_avgOnly){
                    std::cout << _zoomFactor << " Times Covariance of " << _valueName << std::endl << _zoomFactor*_cov << std::endl;
                }
            }
        }
    }
private:
    VecX _exp;
    MatX _cov;
    MatX _defaultWeight;
    bool _avgOnly;
    unsigned int _size;
    unsigned int _measureCount;
    unsigned int _showPeriod;
    unsigned int _waitCount;
    double _zoomFactor;
    std::string _valueName;
};

#endif  // MATHTOOLS_H
