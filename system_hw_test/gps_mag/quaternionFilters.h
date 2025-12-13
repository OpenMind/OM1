#ifndef _QUATERNIONFILTERS_H_
#define _QUATERNIONFILTERS_H_

/* MPU9250 Basic Example Code
 by: Kris Winer
 date: April 1, 2014
 license: Beerware - Use this code however you'd like. If you
 find it useful you can buy me a beer some time.
 Modified by Brent Wilkins July 19, 2016
 Modified by Owen Lyke Aug 6 2018
 - Removed code about magnetometer that would cause an abort()
 - Magnetometer can be accessed by using the MPU9250 as an I2C
   master. See the register map for more details

 Demonstrate basic MPU-9250 functionality including parameterizing the register
 addresses, initializing the sensor, getting properly scaled accelerometer,
 gyroscope, and magnetometer data out. Added display functions to allow display
 to on breadboard monitor. Addition of 9 DoF sensor fusion using open source
 Madgwick and Mahony filter algorithms. Sketch runs on the 3.3 V 8 MHz Pro Mini
 and the Teensy 3.1.
*/

#include <Arduino.h>

void MadgwickQuaternionUpdate(float ax, float ay, float az,
                              float gx, float gy, float gz,
                              float mx, float my, float mz,
                              float deltat);
void MahonyQuaternionUpdate(float ax, float ay, float az,
                            float gx, float gy, float gz,
                            float mx, float my, float mz,
                            float deltat);
const float * getQ();

#endif // _QUATERNIONFILTERS_H_
