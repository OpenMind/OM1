/**********************************************************************
 Copyright (c) 2020-2023, Unitree Robotics.Co.Ltd. All rights reserved.
***********************************************************************/
#ifndef SIMULATED_KEYBOARD_H
#define SIMULATED_KEYBOARD_H

#include "interface/CmdPanel.h"
#include <thread>
#include <atomic>
#include <cmath>

// Simulated KeyBoard class to directly assign values to userCmd and userValue
class SimulatedKeyBoard : public CmdPanel {
public:
    SimulatedKeyBoard();
    ~SimulatedKeyBoard();

protected:
    void* run(void *arg) override;

private:
    // Private methods
    void simulateInput();

    // Threading and control
    std::atomic<bool> running; // Controls the simulation thread
    std::thread inputThread;   // Thread to simulate input
};

#endif  // SIMULATED_KEYBOARD_H
