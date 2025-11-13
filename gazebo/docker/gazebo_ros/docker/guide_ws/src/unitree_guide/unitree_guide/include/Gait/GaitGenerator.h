/**********************************************************************
 Copyright (c) 2020-2023, Unitree Robotics.Co.Ltd. All rights reserved.
***********************************************************************/
#ifndef GAITGENERATOR_H
#define GAITGENERATOR_H

#include "Gait/WaveGenerator.h"
#include "Gait/FeetEndCal.h"

#ifdef COMPILE_DEBUG
#include <common/PyPlot.h>
#endif  // COMPILE_DEBUG

/*cycloid gait*/
class GaitGenerator{
public:
    GaitGenerator(CtrlComponents *ctrlComp);
    ~GaitGenerator();
    void setGait(Vec2 vxyGoalGlobal, float dYawGoal, float gaitHeight);
    void run(Vec34 &feetPos, Vec34 &feetVel);
    Vec3 getFootPos(int i);
    Vec3 getFootVel(int i);
    void restart();
private:
    float cycloidXYPosition(float startXY, float endXY, float phase);
    float cycloidXYVelocity(float startXY, float endXY, float phase);
    float cycloidZPosition(float startZ, float height, float phase);
    float cycloidZVelocity(float height, float phase);

    WaveGenerator *_waveG;
    Estimator *_est;
    FeetEndCal *_feetCal;
    QuadrupedRobot *_robModel;
    LowlevelState *_state;
    float _gaitHeight;
    Vec2 _vxyGoal;
    float _dYawGoal;
    Vec4 *_phase, _phasePast;
    VecInt4 *_contact;
    Vec34 _startP, _endP, _idealP, _pastP;
    bool _firstRun;

#ifdef COMPILE_DEBUG
    PyPlot _testGaitPlot;
#endif  // COMPILE_DEBUG

};

#endif  // GAITGENERATOR_H
