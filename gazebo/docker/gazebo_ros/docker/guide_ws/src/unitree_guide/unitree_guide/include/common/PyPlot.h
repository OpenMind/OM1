/**********************************************************************
 Copyright (c) 2020-2023, Unitree Robotics.Co.Ltd. All rights reserved.
***********************************************************************/
#ifndef PYPLOT_H
#define PYPLOT_H

#include <map>
#include <thread>
#include <vector>
#include <string>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include "thirdParty/matplotlibcpp.h"
#include "common/timeMarker.h"

namespace plt = matplotlibcpp;

struct Curve{
    std::vector<double> x;
    std::vector<double> y;

    void printXY(double xRough, int pointNum){
        for(int i(0); i<x.size(); ++i){
            if(xRough < x[i]){
                for(int j(0); j < pointNum; ++j){
                    std::cout << "  X: " << x[i+j] << ", Y: " << y[i+j] << std::endl;
                }
                break;
            }
        }
    }
};

struct Plot{
    std::vector<Curve*> curves;
    std::vector<std::string> labels;
    std::string plotName;
    int curveCount;

    std::map<std::string, int> curveName2ID;

    Plot(std::string name, int count, std::vector<std::string> labelVec)
        :plotName(name), curveCount(count), labels(labelVec){
        for(int i(0); i < count; ++i){
            curveName2ID.insert(std::pair<std::string, int>(labels[i], i));
            curves.push_back(new Curve());
        }
    }

    ~Plot(){
        for(int i(0); i < curveCount; ++i){
            delete curves[i];
        }
    }

    double getX(long long startT){
        return (double)(getSystemTime() - startT) * 1e-6;
    }

    void printXY(std::string curveName, double xRough, int pointNum){
        std::cout << "[DEBUG] Plot: " << plotName << ", Curve: " << curveName << std::endl;
        curves[curveName2ID[curveName]]->printXY(xRough, pointNum);
    }
};

class PyPlot{
public:
    PyPlot();
    ~PyPlot();
    void addPlot(std::string plotName, int curveCount, std::vector<std::string> labelVec);
    void addPlot(std::string plotName, int curveCount);
    void showPlot(std::string plotName);
    void showPlot(std::vector<std::string> plotNameVec);
    void showPlotAll();

    void printXY(std::string plotName, std::string curveName, double xRough, int pointNum = 1);

    void addFrame(std::string plotName, double value);
    void addFrame(std::string plotName, double x, double value);

    template <typename T>
    void addFrame(std::string plotName, T* valueArray);
    template <typename T>
    void addFrame(std::string plotName, double x, T* valueArray);

    template <typename T>
    void addFrame(std::string plotName, const Eigen::MatrixBase<T> &vec);
    template <typename T>
    void addFrame(std::string plotName, double x, const Eigen::MatrixBase<T> &vec);

    template <typename T>
    void addFrame(std::string plotName, const std::vector<T> &vec);
    template <typename T>
    void addFrame(std::string plotName, double x, const std::vector<T> &vec);

private:
    void _checkStart();
    int _plotCount = 0;
    std::map<std::string, int> _plotName2ID;
    std::vector< Plot* > _plots;
    long long _pointNum;
    Plot* _getPlotPtr(std::string plotName);
    bool start;
    long long startT;
};

inline PyPlot::PyPlot(){
    start = false;
}

inline PyPlot::~PyPlot(){
    for(int i(0); i < _plotCount; ++i){
        delete _plots[i];
    }
}

inline void PyPlot::_checkStart(){
    if(!start){
        start = true;
        startT = getSystemTime();
    }
}

inline void PyPlot::printXY(std::string plotName, std::string curveName, double xRough, int pointNum){
    _plots[_plotName2ID[plotName]]->printXY(curveName, xRough, pointNum);
}

inline void PyPlot::addPlot(std::string plotName, int curveCount, std::vector<std::string> labelVec){
    if(_plotName2ID.count(plotName) == 0){
        _plotName2ID.insert(std::pair<std::string, int>(plotName, _plotCount));
        ++_plotCount;

        _plots.push_back( new Plot(plotName, curveCount, labelVec) );
    }else{
        std::cout << "[ERROR] Already has same Plot: " << plotName << std::endl;
        exit(-1);
    }
}

inline void PyPlot::addPlot(std::string plotName, int curveCount){
    std::vector<std::string> label;
    for(int i(0); i < curveCount; ++i){
        label.push_back(std::to_string(i+1));
    }
    addPlot(plotName, curveCount, label);
}

inline void PyPlot::showPlot(std::string plotName){
    Plot* plot = _getPlotPtr(plotName);
    plt::figure();
    plt::title(plot->plotName);
    for(int i(0); i < plot->curveCount; ++i){
        plt::named_plot(plot->labels[i], plot->curves[i]->x, plot->curves[i]->y);
    }
    plt::legend();
    plt::show();
}

inline void PyPlot::showPlot(std::vector<std::string> plotNameVec){
    for(std::vector<std::string>::iterator itName = plotNameVec.begin(); itName != plotNameVec.end(); ++itName){
        plt::figure();
        Plot* plot = _plots[_plotName2ID[*itName]];
        plt::title(plot->plotName);
        for(int i(0); i < plot->curveCount; ++i){
            plt::named_plot(plot->labels[i], plot->curves[i]->x, plot->curves[i]->y);
        }
        plt::legend();
    }
    plt::show();
}

inline void PyPlot::showPlotAll(){
    for(int i(0); i < _plotCount; ++i){
        plt::figure();
        Plot* plot = _plots[i];
        plt::title(plot->plotName);
        for(int j(0); j < plot->curveCount; ++j){
            plt::named_plot(plot->labels[j], plot->curves[j]->x, plot->curves[j]->y);
        }
        plt::legend();
    }
    plt::show();
    exit(0);
}

inline Plot* PyPlot::_getPlotPtr(std::string plotName){
    if(_plotName2ID.count(plotName) == 0){
        std::cout << "[ERROR] Plot " << plotName << " does not exist" << std::endl;
        exit(-1);
    }else{
        return _plots[_plotName2ID[plotName]];
    }
}

inline void PyPlot::addFrame(std::string plotName, double value){
    _checkStart();
    Plot* plot = _getPlotPtr(plotName);
    addFrame(plotName, plot->getX(startT), value);
}

inline void PyPlot::addFrame(std::string plotName, double x, double value){
    Plot* plot = _getPlotPtr(plotName);

    plot->curves[0]->x.push_back(x);
    plot->curves[0]->y.push_back(value);
}

template <typename T>
inline void PyPlot::addFrame(std::string plotName, T* valueArray){
    _checkStart();
    Plot* plot = _getPlotPtr(plotName);
    addFrame(plotName, plot->getX(startT), valueArray);
}

template <typename T>
inline void PyPlot::addFrame(std::string plotName, double x, T* valueArray){
    Plot* plot = _getPlotPtr(plotName);

    for(int i(0); i < plot->curveCount; ++i){
        plot->curves[i]->x.push_back(x);
        plot->curves[i]->y.push_back(valueArray[i]);
    }
}

template <typename T>
inline void PyPlot::addFrame(std::string plotName, const Eigen::MatrixBase<T> &vec){
    _checkStart();
    Plot* plot = _getPlotPtr(plotName);
    addFrame(plotName, plot->getX(startT), vec);
}

template <typename T>
inline void PyPlot::addFrame(std::string plotName, double x, const Eigen::MatrixBase<T> &vec){
    Plot* plot = _getPlotPtr(plotName);

    for(int i(0); i < plot->curveCount; ++i){
        plot->curves[i]->x.push_back(x);
        plot->curves[i]->y.push_back(vec(i));
    }
}

template <typename T>
inline void PyPlot::addFrame(std::string plotName, const std::vector<T> &vec){
    _checkStart();
    Plot* plot = _getPlotPtr(plotName);
    addFrame(plotName, plot->getX(startT), vec);
}

template <typename T>
inline void PyPlot::addFrame(std::string plotName, double x, const std::vector<T> &vec){
    Plot* plot = _getPlotPtr(plotName);

    for(int i(0); i < plot->curveCount; ++i){
        plot->curves[i]->x.push_back(x);
        plot->curves[i]->y.push_back(vec[i]);
    }
}
#endif // PYPLOT_H
