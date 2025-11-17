/************************************************************************
Copyright (c) 2020, Unitree Robotics.Co.Ltd. All rights reserved.
Use of this source code is governed by the MPL-2.0 license, see LICENSE.
************************************************************************/

#ifndef _UNITREE_LEGGED_LOOP_H_
#define _UNITREE_LEGGED_LOOP_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <thread>
#include <pthread.h>
#include <vector>
#include <boost/shared_ptr.hpp>
#include <boost/function.hpp>

namespace UNITREE_LEGGED_SDK
{

constexpr int THREAD_PRIORITY    = 99;   // real-time priority

typedef boost::function<void ()> Callback;

class Loop {
public:
  Loop(std::string name, float period, int bindCPU = -1):_name(name), _period(period), _bindCPU(bindCPU) {}
  ~Loop();
  void start();
  void shutdown();
  virtual void functionCB() = 0;

private:
  void entryFunc();

  std::string _name;
  float _period;
  int _bindCPU;
  bool _bind_cpu_flag = false;
  bool _isrunning = false;
  std::thread _thread;
};

class LoopFunc : public Loop {
public:
  LoopFunc(std::string name, float period, const Callback& _cb)
    : Loop(name, period), _fp(_cb){}
  LoopFunc(std::string name, float period, int bindCPU, const Callback& _cb)
    : Loop(name, period, bindCPU), _fp(_cb){}
  void functionCB() { (_fp)(); }
private:
  boost::function<void ()>  _fp;
};


}

#endif
