/**********************************************************************
 Copyright (c) 2020-2023, Unitree Robotics.Co.Ltd. All rights reserved.
***********************************************************************/
#ifndef LOWPASSFILTER
#define LOWPASSFILTER

class LPFilter{
public:
    LPFilter(double samplePeriod, double cutFrequency);
    ~LPFilter();
    void addValue(double newValue);
    double getValue();
    void clear();
private:
    double _weight;
    double _pastValue;
    bool _start;
};

#endif  // LOWPASSFILTER
