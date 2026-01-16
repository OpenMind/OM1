// for the GPS
#include <Adafruit_GPS.h>
#define GPSSerial Serial1
Adafruit_GPS GPS(&GPSSerial);

// PMTK commands to turn on SBAS/WAAS (DGPS)
#define PMTK_ENABLE_SBAS   "$PMTK313,1*2E"  // enable SBAS search
#define PMTK_ENABLE_WAAS   "$PMTK301,2*2E"  // set SBAS mode = WAAS

// for the mag
#include <Wire.h>
#include <Adafruit_LIS3MDL.h>
#include <Adafruit_Sensor.h>
Adafruit_LIS3MDL lis3mdl;

#include <Adafruit_LSM6DS3TRC.h>
Adafruit_LSM6DS3TRC lsm6ds3trc;

uint32_t timerGPS = millis();
uint32_t timerMAG = millis();

int IMU_LAST_READ = 0;

/* For calibration, set CALIBRATION to true, run the script, close
the Arduino IDE, close the serial monitor, and open MotionCal.

MotionCal can be downloaded here:

https://github.com/PaulStoffregen/MotionCal

If you do not see your Arduino in MotionCal's serial port
drop down menu, follow these instructions to compile MotionCal
for your OS and platform

https://github.com/PaulStoffregen/MotionCal/issues/11#issuecomment-2412937251
*/

bool CALIBRATION = false;

float ToDecimalDegrees(float formattedLatLon)
{
  float decDegrees = (float)((int)formattedLatLon / 100);
  float decMinutes = formattedLatLon - (decDegrees * 100);
  float fractDegrees = decMinutes / 60.0;

  return decDegrees + fractDegrees;
}

void PrintFloatArray(float ar[], int n)
{
  for(int i = 0; i < n; i++) {
    Serial.print(ar[i]);
    if(i == n - 1) {
      Serial.print("\n");
    } else {
      Serial.print(":");
    }
  }
}

// These are specific to each IMU and IMU environment -
// Poor calibration values will prevent AHRS function
// Add your device specific values below, in the init loop
float mag_hardiron[]  = { 0.0, 0.0, 0.0 }; // in uTesla
float mag_softiron[]  = { 1.0, 0.0, 0.0, \
                          0.0, 1.0, 0.0, \
                          0.0, 0.0, 1.0 };
float gyro_zerorate[] = { 0.0, 0.0, 0.0 }; // in Radians/s

//declination = +12° 50'

#include "quaternionFilters.h"
#define MPS2_TO_G 9.80665

void setup()
{
  Serial.begin(115200);
  while (!Serial) delay(10);

  // print device ID and pick calibration set...
  char DeviceID[9];
  itoa(NRF_FICR->DEVICEID[0], DeviceID, 16);
  Serial.print("Device ID 0: ");
  Serial.println(DeviceID);

  if (strcmp(DeviceID, "38e4bfd6") == 0) {
    float mh[]  = { 22.84, -11.84, 3.96 }; // in uTesla
    float ms[]  = { +0.951, +0.059, -0.031, \
                    +0.059, +1.091, +0.001, \
                    -0.031, +0.001, +0.968 };
    float gc[] = { 0.000,  0.020,  0.010}; // in Radians/s
    memcpy(mag_hardiron,  mh, sizeof(mag_hardiron));
    memcpy(mag_softiron,  ms, sizeof(mag_softiron));
    memcpy(gyro_zerorate, gc, sizeof(gyro_zerorate));
  } else if (strcmp(DeviceID, "cfcdd0fb") == 0) {
    // not sure which Arduino this is
    float mh[]  = { -15.53, 5.38, 19.17 }; // in uTesla
    float ms[]  = { 0.934, 0.048, -0.021, \
                              0.048, 1.090, 0.003, \
                              -0.021, 0.003, 0.985 };
    float gc[] = { 0.0, 0.0, 0.0 };
    memcpy(mag_hardiron,  mh, sizeof(mag_hardiron));
    memcpy(mag_softiron,  ms, sizeof(mag_softiron));
    memcpy(gyro_zerorate, gc, sizeof(gyro_zerorate));
  } else if (strcmp(DeviceID, "ec414ffd") == 0) {
    float mh[]  = { +12.26, -23.74, -9.67 }; // in uTesla
    float ms[]  = { +0.948, +0.046, -0.038, \
                    +0.047, +1.087, -0.033, \
                    -0.036, -0.033, +0.974 };
    float gc[] = { -0.0055, 0.0111, 0.0017 };
    memcpy(mag_hardiron,  mh, sizeof(mag_hardiron));
    memcpy(mag_softiron,  ms, sizeof(mag_softiron));
    memcpy(gyro_zerorate, gc, sizeof(gyro_zerorate));
  } else if (strcmp(DeviceID, "c80e5b02") == 0) {
    float mh[]  = { -24.32, +4.52, +12.20 }; // in uTesla
    float ms[]  = { +0.954, +0.061, -0.033, \
                    +0.061, +1.085, +0.002, \
                    -0.033, +0.002, +0.969 };
    float gc[] = { -0.0328, +0.0521, 0.0156 };
    memcpy(mag_hardiron,  mh, sizeof(mag_hardiron));
    memcpy(mag_softiron,  ms, sizeof(mag_softiron));
    memcpy(gyro_zerorate, gc, sizeof(gyro_zerorate));
  }
  else {
    Serial.println("CAUTION: Magnetometer not calibrated - code will yield garbage - please calibrate your Magnetometer and IMU");
  }

  Serial.println("Using these calibration values:");
  PrintFloatArray(mag_hardiron,  3);
  PrintFloatArray(mag_softiron,  9);
  PrintFloatArray(gyro_zerorate, 3);

  delay(5000);

  // --- GPS init ---
  Serial.println("GPS booting");
  GPS.begin(9600);
  GPS.sendCommand(PMTK_SET_NMEA_OUTPUT_RMCGGA);
  GPS.sendCommand(PMTK_SET_NMEA_UPDATE_1HZ);
  GPS.sendCommand(PGCMD_ANTENNA);

  // ==== DGPS enable ====
  GPS.sendCommand(PMTK_ENABLE_SBAS);
  GPS.sendCommand(PMTK_ENABLE_WAAS);
  // ======================

  delay(1000);

  // Ask for firmware version
  GPSSerial.println(PMTK_Q_RELEASE);

  // --- Magnetometer init ---
  Serial.println("Booting LIS3MDL");

  if (!lis3mdl.begin_I2C()) {
    Serial.println("Failed to find LIS3MDL");
    while (1) { delay(10); }
  }
  Serial.println("LIS3MDL Found");

  lis3mdl.setPerformanceMode(LIS3MDL_MEDIUMMODE);
  lis3mdl.setOperationMode(LIS3MDL_CONTINUOUSMODE);
  lis3mdl.setDataRate(LIS3MDL_DATARATE_155_HZ);
  lis3mdl.setRange(LIS3MDL_RANGE_4_GAUSS);
  lis3mdl.setIntThreshold(500);
  lis3mdl.configInterrupt(false, false, true, // enable z axis
                          true,  // polarity
                          false, // don't latch
                          true); // enabled!

  Serial.println("Booting LSM6DS3TR-C");

  if (!lsm6ds3trc.begin_I2C()) {
    Serial.println("Failed to find LSM6DS3TR-C");
    while (1) { delay(10); }
  }
  Serial.println("LSM6DS3TR-C Found");

  lsm6ds3trc.setAccelRange(LSM6DS_ACCEL_RANGE_2_G);
  lsm6ds3trc.setGyroRange(LSM6DS_GYRO_RANGE_250_DPS);
  lsm6ds3trc.setAccelDataRate(LSM6DS_RATE_12_5_HZ);
  lsm6ds3trc.setGyroDataRate(LSM6DS_RATE_12_5_HZ);
  lsm6ds3trc.configInt1(false, false, true); // accelerometer DRDY on INT1
  lsm6ds3trc.configInt2(false, true, false); // gyro DRDY on INT2
}

void loop()
{

  char c = GPS.read();
  // if a sentence is received, we can check the checksum, parse it...
  if (GPS.newNMEAreceived()) {
    if (!GPS.parse(GPS.lastNMEA())) // this also sets the newNMEAreceived() flag to false
      return; // we can fail to parse a sentence in which case we should just wait for another
  }

  // approximately every 1 seconds or so, print out the current stats
  if (millis() - timerGPS > 1000) {
    timerGPS = millis(); // reset the timer

    if (GPS.fix && CALIBRATION == false) {

      float latDecimal = ToDecimalDegrees(GPS.latitude);
      float lonDecimal = ToDecimalDegrees(GPS.longitude);

      // use this if you want DDMM.MMMM and DDDMM.MMMM
      // Degrees and minutes
      // Serial.print("GPS:");
      // Serial.print(GPS.latitude, 4);
      // Serial.print(GPS.lat);
      // Serial.print(",");
      // Serial.print(GPS.longitude, 4);
      // Serial.print(GPS.lon);

      // use this if you want decimal degrees
      Serial.print("GPS:");
      Serial.print(latDecimal,8);
      Serial.print(GPS.lat);
      Serial.print(",");
      Serial.print(lonDecimal,8);
      Serial.print(GPS.lon);

      Serial.print(",KN:");
      Serial.print(GPS.speed);
      Serial.print(",HEAD:");
      Serial.print(GPS.angle);
      Serial.print(",ALT:");
      Serial.print(GPS.altitude);
      Serial.print(",SAT:");
      Serial.print((int)GPS.satellites);
      Serial.print(",TIME:");
      Serial.print(GPS.hour);
      Serial.print(":");
      Serial.print(GPS.minute);
      Serial.print(":");
      Serial.print(GPS.seconds);
      Serial.print(":");
      Serial.println(GPS.milliseconds);
    } else {
      Serial.print("STAT: Waiting for GPS fix. SAT:");
      Serial.println((int)GPS.satellites);
    }
  }

  // approximately 500 ms or so, update the direction
  if (millis() - timerMAG > 50) {
    timerMAG = millis();
    float imu_dt_s = float(timerMAG - IMU_LAST_READ) / 1000;
    IMU_LAST_READ = timerMAG;

    // Get a new normalized sensor event
    sensors_event_t accel;
    sensors_event_t gyro;
    sensors_event_t mag;
    sensors_event_t temp;

    lsm6ds3trc.getEvent(&accel, &gyro, &temp);
    lis3mdl.getEvent(&mag);

    if (CALIBRATION) {
      float ax_G = accel.acceleration.x / MPS2_TO_G;
      float ay_G = accel.acceleration.y / MPS2_TO_G;
      float az_G = accel.acceleration.z / MPS2_TO_G;

      float gx = gyro.gyro.x;
      float gy = gyro.gyro.y;
      float gz = gyro.gyro.z;

      // hard iron cal
      float mx = mag.magnetic.x;
      float my = mag.magnetic.y;
      float mz = mag.magnetic.z;

      Serial.print("Raw:");
      Serial.print(int(ax_G*10000)); Serial.print(",");
      Serial.print(int(ay_G*10000)); Serial.print(",");
      Serial.print(int(az_G*10000)); Serial.print(",");
      Serial.print(int(gx*RAD_TO_DEG)); Serial.print(","); // Deg resolution, that's ok
      Serial.print(int(gy*RAD_TO_DEG)); Serial.print(",");
      Serial.print(int(gz*RAD_TO_DEG)); Serial.print(",");
      Serial.print(int(mx*10)); Serial.print(","); // uT*10
      Serial.print(int(my*10)); Serial.print(",");
      Serial.print(int(mz*10)); Serial.println("");

      // unified data
      Serial.print("Uni:");
      Serial.print(ax_G); Serial.print(",");
      Serial.print(ay_G); Serial.print(",");
      Serial.print(az_G); Serial.print(",");
      Serial.print(gx, 4); Serial.print(",");
      Serial.print(gy, 4); Serial.print(",");
      Serial.print(gz, 4); Serial.print(",");
      Serial.print(mx); Serial.print(",");
      Serial.print(my); Serial.print(",");
      Serial.print(mz); Serial.println("");

      return;
    }

    // Serial.print("T:");
    // Serial.print(temp.temperature);
    // Serial.println("C");

    // /* Display the results (acceleration is measured in m/s^2) */
    // Serial.print("AX:");
    // Serial.print(accel.acceleration.x);
    // Serial.print("Y:");
    // Serial.print(accel.acceleration.y);
    // Serial.print("Z:");
    // Serial.print(accel.acceleration.z);
    // Serial.println(" m/s^2");

    // /* Display the results (rotation is measured in rad/s) */
    // Serial.print("GX:");
    // Serial.print(gyro.gyro.x);
    // Serial.print("Y:");
    // Serial.print(gyro.gyro.y);
    // Serial.print("Z:");
    // Serial.print(gyro.gyro.z);
    // Serial.println(" rad/s");

    // /* Display the results (magnetic field is measured in uTesla) */
    // Serial.print("MAG:X:"); Serial.print(mag.magnetic.x);
    // Serial.print("Y:"); Serial.print(mag.magnetic.y);
    // Serial.print("Z:"); Serial.print(mag.magnetic.z);
    // Serial.println("uT");

    // Serial.print("DT(s):");
    // Serial.println(imu_dt_s);

    float ax_G = accel.acceleration.x / MPS2_TO_G;
    float ay_G = accel.acceleration.y / MPS2_TO_G;
    float az_G = accel.acceleration.z / MPS2_TO_G;

    float gx = gyro.gyro.x + gyro_zerorate[0];
    float gy = gyro.gyro.y + gyro_zerorate[1];
    float gz = gyro.gyro.z + gyro_zerorate[2];

    // hard iron cal
    float mx = mag.magnetic.x - mag_hardiron[0];
    float my = mag.magnetic.y - mag_hardiron[1];
    float mz = mag.magnetic.z - mag_hardiron[2];

    // soft iron cal
    float mxCal = mx * mag_softiron[0] + my * mag_softiron[1] + mz * mag_softiron[2];
    float myCal = mx * mag_softiron[3] + my * mag_softiron[4] + mz * mag_softiron[5];
    float mzCal = mx * mag_softiron[6] + my * mag_softiron[7] + mz * mag_softiron[8];

    MahonyQuaternionUpdate(ax_G, ay_G, az_G, gx, gy, gz, mxCal, myCal, mzCal, imu_dt_s);

    // String dataLog = String("DATA:");
    // dataLog += String(" AX:") + String(ax_G) + String(",AY:") + String(ay_G) + String(",AZ:") + String(az_G);
    // dataLog += String(",GX:") + String(gx) + String(",GY:") + String(gy) + String(",GZ:") + String(gz);
    // dataLog += String(",MX:") + String(mxCal) + String(",MY:") + String(myCal) + String(",MZ:") + String(mzCal);

    // Serial.println(dataLog);

    // NOTE: Written by Kris Winer
    // Define output variables from updated quaternion---these are Tait-Bryan
    // angles, commonly used in aircraft orientation. In this coordinate system,
    // the positive z-axis is down toward Earth. Yaw is the angle between Sensor
    // x-axis and Earth magnetic North (or true North if corrected for local
    // declination, looking down on the sensor positive yaw is counterclockwise.
    // Pitch is angle between sensor x-axis and Earth ground plane, toward the
    // Earth is positive, up toward the sky is negative. Roll is angle between
    // sensor y-axis and Earth ground plane, y-axis up is positive roll. These
    // arise from the definition of the homogeneous rotation matrix constructed
    // from quaternions. Tait-Bryan angles as well as Euler angles are
    // non-commutative; that is, the get the correct orientation the rotations
    // must be applied in the correct order which for this configuration is yaw,
    // pitch, and then roll.
    // For more see
    // http://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    // which has additional links.
    float myIMUyaw = atan2(2.0f * (*(getQ()+1) * *(getQ()+2) + *getQ()
                  * *(getQ()+3)), *getQ() * *getQ() + *(getQ()+1)
                  * *(getQ()+1) - *(getQ()+2) * *(getQ()+2) - *(getQ()+3)
                  * *(getQ()+3));
    float myIMUpitch = -asin(2.0f * (*(getQ()+1) * *(getQ()+3) - *getQ()
                  * *(getQ()+2)));
    float myIMUroll = atan2(2.0f * (*getQ() * *(getQ()+1) + *(getQ()+2)
                  * *(getQ()+3)), *getQ() * *getQ() - *(getQ()+1)
                  * *(getQ()+1) - *(getQ()+2) * *(getQ()+2) + *(getQ()+3)
                  * *(getQ()+3));

    Serial.print("YPR: ");
    Serial.print(myIMUyaw *= RAD_TO_DEG, 2);
    Serial.print(", ");
    Serial.print(myIMUpitch *= RAD_TO_DEG, 2);
    Serial.print(", ");
    Serial.println(myIMUroll *= RAD_TO_DEG, 2);

  float headingDegrees = 0.0;
  if (myIMUyaw < 0) {
    headingDegrees = -1.0 * myIMUyaw;
  } else {
    headingDegrees = -1.0 * myIMUyaw + 360.0;
  }

  String cardinal;
  if (headingDegrees > 348.75 || headingDegrees < 11.25) {
    cardinal = " N";
  }
  else if (headingDegrees > 11.25 && headingDegrees < 33.75) {
    cardinal = " NNE";
  }
  else if (headingDegrees > 33.75 && headingDegrees < 56.25) {
    cardinal = " NE";
  }
  else if (headingDegrees > 56.25 && headingDegrees < 78.75) {
    cardinal = " ENE";
  }
  else if (headingDegrees > 78.75 && headingDegrees < 101.25) {
    cardinal = " E";
  }
  else if (headingDegrees > 101.25 && headingDegrees < 123.75) {
    cardinal = " ESE";
  }
  else if (headingDegrees > 123.75 && headingDegrees < 146.25) {
    cardinal = " SE";
  }
  else if (headingDegrees > 146.25 && headingDegrees < 168.75) {
    cardinal = " SSE";
  }
  else if (headingDegrees > 168.75 && headingDegrees < 191.25) {
    cardinal = " S";
  }
  else if (headingDegrees > 191.25 && headingDegrees < 213.75) {
    cardinal = " SSW";
  }
  else if (headingDegrees > 213.75 && headingDegrees < 236.25) {
    cardinal = " SW";
  }
  else if (headingDegrees > 236.25 && headingDegrees < 258.75) {
    cardinal = " WSW";
  }
  else if (headingDegrees > 258.75 && headingDegrees < 281.25) {
    cardinal = " W";
  }
  else if (headingDegrees > 281.25 && headingDegrees < 303.75) {
    cardinal = " WNW";
  }
  else if (headingDegrees > 303.75 && headingDegrees < 326.25) {
    cardinal = " NW";
  }
  else if (headingDegrees > 326.25 && headingDegrees < 348.75) {
    cardinal = " NNW";
  }

  float headingRadiansNTC = atan2(myCal, mxCal);
  float headingDegreesNTC = headingRadiansNTC * 180 / PI;
  if (headingDegreesNTC < 0) {
    headingDegreesNTC += 360;
  }

  Serial.print("HDG (DEG): ");
  Serial.print(headingDegrees);
  Serial.print(cardinal);
  Serial.print(" NTC_HDG: ");
  Serial.println(headingDegreesNTC);
  }
}
