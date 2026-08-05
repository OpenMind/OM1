package bme680

import (
	"math"
	"os"
)

// calibration holds the factory calibration coefficients read from the sensor's
// NVM registers, per the Bosch BME680 datasheet (section 3.4 "Trimming
// coefficients").
type calibration struct {
	// Temperature
	parT1 uint16
	parT2 int16
	parT3 int8

	// Humidity
	parH1, parH2 uint16
	parH3, parH4, parH5, parH6, parH7 int8

	// Gas
	parG1 int8
	parG2 int16
	parG3 int8
	resHeatRange byte
	resHeatVal   int8
	rangeSwErr   int8
}

// readCalibration reads and parses both calibration blocks from the sensor.
func readCalibration(f *os.File) (calibration, error) {
	var c calibration

	b1, err := readBlock(f, regCalibStart1, 25) // 0x89-0xA1
	if err != nil {
		return c, err
	}
	b2, err := readBlock(f, regCalibStart2, 16) // 0xE1-0xF0
	if err != nil {
		return c, err
	}

	u16 := func(lsb, msb byte) uint16 { return uint16(msb)<<8 | uint16(lsb) }
	s16 := func(lsb, msb byte) int16 { return int16(u16(lsb, msb)) }

	// Offsets below follow the Bosch datasheet register map for calibration data.
	c.parT2 = s16(b1[1], b1[2])
	c.parT3 = int8(b1[3])
	c.parT1 = u16(b2[8], b2[9])

	c.parH2 = u16(b2[1], b2[0]) >> 4 // note: reversed byte order per datasheet
	c.parH1 = u16(b2[2], b2[1]) & 0x0FFF
	c.parH3 = int8(b2[3])
	c.parH4 = int8(b2[4])
	c.parH5 = int8(b2[5])
	c.parH6 = int8(b2[6])
	c.parH7 = int8(b2[7])

	c.parG2 = s16(b1[20], b1[21])
	c.parG1 = int8(b1[22])
	c.parG3 = int8(b1[23])

	rh, err := readReg(f, regHeatRange)
	if err != nil {
		return c, err
	}
	c.resHeatRange = (rh >> 4) & 0x03

	hv, err := readReg(f, regHeatVal)
	if err != nil {
		return c, err
	}
	c.resHeatVal = int8(hv)

	se, err := readReg(f, regRangeSwErr)
	if err != nil {
		return c, err
	}
	c.rangeSwErr = int8(se>>4) & 0x0F

	return c, nil
}

// compensateTemp implements the datasheet's temperature compensation formula.
// Returns (t_fine, temperature in °C) — t_fine is needed by humidity compensation.
func (c calibration) compensateTemp(adcT int32) (int32, float64) {
	var1 := (float64(adcT)/16384.0 - float64(c.parT1)/1024.0) * float64(c.parT2)
	var2 := math.Pow(float64(adcT)/131072.0-float64(c.parT1)/8192.0, 2) * float64(c.parT3) * 16.0
	tFine := int32(var1 + var2)
	tempC := (var1 + var2) / 5120.0
	return tFine, tempC
}

// compensateHumidity implements the datasheet's humidity compensation formula.
func (c calibration) compensateHumidity(adcH int32, tFine int32) float64 {
	tempComp := float64(tFine) / 5120.0

	var1 := float64(adcH) - (float64(c.parH1)*16.0 + (float64(c.parH3)/2.0)*tempComp)
	var2 := var1 * (float64(c.parH2) / 262144.0 * (1.0 + (float64(c.parH4)/16384.0)*tempComp + (float64(c.parH5)/1048576.0)*tempComp*tempComp))
	var3 := float64(c.parH6) / 16384.0
	var4 := float64(c.parH7) / 2097152.0
	humidity := var2 + (var3+var4*tempComp)*var2*var2

	if humidity > 100 {
		humidity = 100
	}
	if humidity < 0 {
		humidity = 0
	}
	return humidity
}

// compensateGas implements the datasheet's gas resistance compensation formula.
func (c calibration) compensateGas(adcGas uint16, gasRange byte) float64 {
	lookupK1 := [16]float64{
		1, 1, 1, 1, 1, 0.99, 1, 0.992,
		1, 1, 0.998, 0.995, 1, 0.99, 1, 1,
	}
	lookupK2 := [16]float64{
		8000000, 4000000, 2000000, 1000000, 499500.4995, 248262.1648, 125000, 63000.03938,
		31281.28128, 15625, 7812.5, 3906.25, 1953.125, 976.5625, 488.28125, 244.140625,
	}

	varGasSwitching := lookupK1[gasRange]*1340.0 + 5.0
	var1 := varGasSwitching * (1340.0 + 5.0*float64(c.rangeSwErr)) / 100.0
	var2 := var1 * (float64(adcGas)*1.0 - 512.0 + var1)
	gasResistance := lookupK2[gasRange] * var1 / var2
	return gasResistance
}

// calcHeaterResistance calculates the res_heat_x register value for a target
// heater plate temperature (°C), given ambient temperature (°C), per datasheet.
func (c calibration) calcHeaterResistance(targetTempC, ambientTempC float64) byte {
	var1 := float64(c.parG1)/16.0 + 49.0
	var2 := (float64(c.parG2)/32768.0)*0.0005 + 0.00235
	var3 := float64(c.parG3) / 1024.0
	var4 := var1 * (1.0 + var2*targetTempC)
	var5 := var4 + var3*ambientTempC
	resHeatX := 3.4*(var5*(4.0/(4.0+float64(c.resHeatRange)))*(1.0/(1.0+float64(c.resHeatVal)*0.002))-25.0)
	return byte(resHeatX)
}

func (c *Connector) calcHeaterResistance(targetTempC, ambientTempC float64) byte {
	return c.calib.calcHeaterResistance(targetTempC, ambientTempC)
}

// calcGasWait converts milliseconds into the gas_wait_x register encoding
// (multiplier factor + 4x multiplier bits), per datasheet section 5.3.3.
func calcGasWait(ms int) byte {
	var factor byte
	durVal := ms
	for durVal > 0x3F {
		durVal /= 4
		factor++
	}
	return byte(durVal) | (factor << 6)
}
