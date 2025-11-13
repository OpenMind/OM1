/**********************************************************************
 Copyright (c) 2020-2023, Unitree Robotics.Co.Ltd. All rights reserved.
***********************************************************************/
#include "control/BalanceCtrl.h"
#include "common/mathTools.h"
#include "common/timeMarker.h"

BalanceCtrl::BalanceCtrl(double mass, Mat3 Ib, Mat6 S, double alpha, double beta)
            : _mass(mass), _Ib(Ib), _S(S), _alpha(alpha), _beta(beta){
    _Fprev.setZero();
    _g << 0, 0, -9.81;
    _fricRatio = 0.3;
    _fricMat <<  1,  0, _fricRatio,
                -1,  0, _fricRatio,
                 0,  1, _fricRatio,
                 0, -1, _fricRatio,
                 0,  0, 1;
}

BalanceCtrl::BalanceCtrl(QuadrupedRobot *robModel){
    Vec6 s;
    Vec12 w, u;

    _mass = robModel->getRobMass();
    _pcb = robModel->getPcb();
    _Ib = robModel->getRobInertial();
    _g << 0, 0, -9.81;

    w << 10, 10, 4, 10, 10, 4, 10, 10, 4, 10, 10, 4;
    u << 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3;
    _alpha = 0.001;
    _beta  = 0.1;
    _fricRatio = 0.4;

    s << 20, 20, 50, 450, 450, 450;

    _S = s.asDiagonal();
    _W = w.asDiagonal();
    _U = u.asDiagonal();

    _Fprev.setZero();
    _fricMat <<  1,  0, _fricRatio,
                -1,  0, _fricRatio,
                 0,  1, _fricRatio,
                 0, -1, _fricRatio,
                 0,  0, 1;
}

Vec34 BalanceCtrl::calF(Vec3 ddPcd, Vec3 dWbd, RotMat rotM, Vec34 feetPos2B, VecInt4 contact){
    calMatrixA(feetPos2B, rotM, contact);
    calVectorBd(ddPcd, dWbd, rotM);
    calConstraints(contact);

    _G = _A.transpose()*_S*_A + _alpha*_W + _beta*_U;
    _g0T = -_bd.transpose()*_S*_A - _beta*_Fprev.transpose()*_U;

    solveQP();

    _Fprev = _F;
    return vec12ToVec34(_F);
}

void BalanceCtrl::calMatrixA(Vec34 feetPos2B, RotMat rotM, VecInt4 contact){
    for(int i(0); i < 4; ++i){
        _A.block(0, 3*i, 3, 3) = I3;
        _A.block(3, 3*i, 3, 3) = skew(feetPos2B.col(i) - rotM*_pcb);
    }
}

void BalanceCtrl::calVectorBd(Vec3 ddPcd, Vec3 dWbd, RotMat rotM){
    _bd.head(3) = _mass * (ddPcd - _g);
    _bd.tail(3) = (rotM * _Ib * rotM.transpose()) * dWbd;
}

void BalanceCtrl::calConstraints(VecInt4 contact){
    int contactLegNum = 0;
    for(int i(0); i<4; ++i){
        if(contact(i) == 1){
            contactLegNum += 1;
        }
    }

    _CI.resize(5*contactLegNum, 12);
    _ci0.resize(5*contactLegNum);
    _CE.resize(3*(4 - contactLegNum), 12);
    _ce0.resize(3*(4 - contactLegNum));

    _CI.setZero();
    _ci0.setZero();
    _CE.setZero();
    _ce0.setZero();

    int ceID = 0;
    int ciID = 0;
    for(int i(0); i<4; ++i){
        if(contact(i) == 1){
            _CI.block(5*ciID, 3*i, 5, 3) = _fricMat;
            ++ciID;
        }else{
            _CE.block(3*ceID, 3*i, 3, 3) = I3;
            ++ceID;
        }
    }
}

void BalanceCtrl::solveQP(){
    int n = _F.size();
    int m = _ce0.size();
    int p = _ci0.size();

    G.resize(n, n);
    CE.resize(n, m);
    CI.resize(n, p);
    g0.resize(n);
    ce0.resize(m);
    ci0.resize(p);
    x.resize(n);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            G[i][j] = _G(i, j);
        }
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            CE[i][j] = (_CE.transpose())(i, j);
        }
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < p; ++j) {
            CI[i][j] = (_CI.transpose())(i, j);
        }
    }

    for (int i = 0; i < n; ++i) {
        g0[i] = _g0T[i];
    }

    for (int i = 0; i < m; ++i) {
        ce0[i] = _ce0[i];
    }

    for (int i = 0; i < p; ++i) {
        ci0[i] = _ci0[i];
    }

    double value = solve_quadprog(G, g0, CE, ce0, CI, ci0, x);

    for (int i = 0; i < n; ++i) {
        _F[i] = x[i];
    }
}
