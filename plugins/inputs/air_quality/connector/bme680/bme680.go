package bme680

import (
	"context"
	"fmt"
	"log"
	"math"
	"os"
	"time"

	"golang.org/x/sys/unix"

	"github.com/openmind/om1/plugins/inputs/air_quality/connector"
)

// Register addresses per Bosch BME680 datasheet.
const (
	regChipID       = 0xD0
	regReset        = 0xE0
	regCtrlHum      = 0x72
	regCtrlMeas     = 0x74
	regConfig       = 0x75
	regCtrlGas1     = 0x71
	regGasWait0     = 0x64
	regResHeat0     = 0x5A
	regMeasStatus0  = 0x1D
	regFieldStart   = 0x1F // press_msb..gas_r_lsb block starts here
	regCalibStart1  = 0x89 // calibration block 1 (0x89-0xA1)
	regCalibStart2  = 0xE1 // calibration block 2 (0xE1-0xEF)
	regHeatRange    = 0x02
	regHeatVal      = 0x00
	regRangeSwErr   = 0x04
	chipIDExpected  = 0x61
	i2cSlaveIoctl   = 0x0703 // unix.I2C_SLAVE
)

// Config mirrors the `config: dict` passed into Python's BME680Connector.__init__.
type Config struct {
	I2CBus      string  // e.g. "/dev/i2c-1"
	I2CAddress  uint16  // config["i2c_address"], default 0x76
	Location    string  // config["location"], default "Robot"
	GasBaseline float64 // config["gas_baseline"], default 50000.0
}

// Connector implements connector.AirQualityConnector for the BME680 sensor over I2C.
// Talks to the sensor directly via Linux i2c-dev, following the Bosch BME680
// datasheet register map and compensation formulas (no third-party driver).
type Connector struct {
	cfg    Config
	file   *os.File
	calib  calibration
	logger *log.Logger
}

func New(cfg Config, logger *log.Logger) *Connector {
	if cfg.I2CBus == "" {
		cfg.I2CBus = "/dev/i2c-1"
	}
	if cfg.I2CAddress == 0 {
		cfg.I2CAddress = 0x76
	}
	if cfg.Location == "" {
		cfg.Location = "Robot"
	}
	if cfg.GasBaseline == 0 {
		cfg.GasBaseline = 50000.0
	}
	if logger == nil {
		logger = log.Default()
	}
	return &Connector{cfg: cfg, logger: logger}
}

func (c *Connector) Name() string {
	return "bme680"
}

// Connect opens the I2C bus, verifies chip ID, reads calibration data, and
// configures oversampling + gas heater, mirroring Python's `connect()`.
func (c *Connector) Connect(ctx context.Context) (bool, error) {
	f, err := os.OpenFile(c.cfg.I2CBus, os.O_RDWR, 0)
	if err != nil {
		c.logger.Printf("BME680Connector: failed to connect: %v", err)
		return false, nil
	}

	if err := ioctlSetSlave(f, c.cfg.I2CAddress); err != nil {
		f.Close()
		c.logger.Printf("BME680Connector: failed to connect: %v", err)
		return false, nil
	}

	id, err := readReg(f, regChipID)
	if err != nil || id != chipIDExpected {
		f.Close()
		c.logger.Printf("BME680Connector: unexpected chip id 0x%X (err=%v)", id, err)
		return false, nil
	}

	c.file = f

	calib, err := readCalibration(f)
	if err != nil {
		c.file.Close()
		c.file = nil
		c.logger.Printf("BME680Connector: failed to read calibration: %v", err)
		return false, nil
	}
	c.calib = calib

	// Oversampling: humidity x2, pressure x4, temperature x8 (matches Python defaults).
	if err := writeReg(f, regCtrlHum, 0x02); err != nil {
		return false, nil
	}
	ctrlMeas := byte(0x05<<5) | byte(0x03<<2) // temp OS_8X (101), press OS_4X (011), mode=sleep
	if err := writeReg(f, regCtrlMeas, ctrlMeas); err != nil {
		return false, nil
	}
	// IIR filter size 3 -> filter coeff index 2 (per datasheet table), bits [4:2] of regConfig.
	if err := writeReg(f, regConfig, 0x02<<2); err != nil {
		return false, nil
	}

	// Gas heater: target 320°C for 150ms, profile 0 (matches Python defaults).
	heaterRes := c.calcHeaterResistance(320, 25) // assume 25°C ambient if unknown
	if err := writeReg(f, regResHeat0, heaterRes); err != nil {
		return false, nil
	}
	if err := writeReg(f, regGasWait0, calcGasWait(150)); err != nil {
		return false, nil
	}
	if err := writeReg(f, regCtrlGas1, 0x10); err != nil { // run_gas=1, nb_conv=0 (profile 0)
		return false, nil
	}

	c.logger.Printf("BME680Connector: connected at I2C address 0x%X", c.cfg.I2CAddress)
	return true, nil
}

func (c *Connector) Disconnect(ctx context.Context) error {
	if c.file != nil {
		err := c.file.Close()
		c.file = nil
		c.logger.Println("BME680Connector: disconnected")
		return err
	}
	return nil
}

// Read triggers one forced-mode measurement and parses the result.
// Runs in a goroutine, mirroring Python's `run_in_executor`.
func (c *Connector) Read(ctx context.Context) (*connector.AirQualityData, error) {
	if c.file == nil {
		c.logger.Println("BME680Connector: not connected")
		return nil, nil
	}

	type result struct {
		data *connector.AirQualityData
		err  error
	}
	resCh := make(chan result, 1)

	go func() {
		data, err := c.readSensor()
		if err != nil {
			c.logger.Printf("BME680Connector: read error: %v", err)
			resCh <- result{nil, nil}
			return
		}
		resCh <- result{data, nil}
	}()

	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case res := <-resCh:
		return res.data, res.err
	}
}

// readSensor triggers forced mode, polls for new data, and compensates raw values.
// Mirrors Python's `_read_sensor()`.
func (c *Connector) readSensor() (*connector.AirQualityData, error) {
	// Trigger forced mode: keep OS settings, set mode=01 (forced).
	ctrlMeas, err := readReg(c.file, regCtrlMeas)
	if err != nil {
		return nil, err
	}
	if err := writeReg(c.file, regCtrlMeas, (ctrlMeas&0xFC)|0x01); err != nil {
		return nil, err
	}

	// Poll new_data_0 bit in meas_status_0 (bit 7), timeout ~2s.
	deadline := time.Now().Add(2 * time.Second)
	for {
		status, err := readReg(c.file, regMeasStatus0)
		if err != nil {
			return nil, err
		}
		if status&0x80 != 0 {
			break
		}
		if time.Now().After(deadline) {
			c.logger.Println("BME680Connector: sensor data not ready")
			return nil, nil
		}
		time.Sleep(10 * time.Millisecond)
	}

	field, err := readBlock(c.file, regFieldStart, 8) // press(3)+temp(3)+hum(2)
	if err != nil {
		return nil, err
	}
	gasField, err := readBlock(c.file, 0x2A, 2) // gas_r_msb, gas_r_lsb (contains ADC + range)
	if err != nil {
		return nil, err
	}

	tempADC := int32(field[3])<<12 | int32(field[4])<<4 | int32(field[5])>>4
	humADC := int32(field[6])<<8 | int32(field[7])

	tFine, tempC := c.calib.compensateTemp(tempADC)
	humidity := c.calib.compensateHumidity(humADC, tFine)

	gasADC := uint16(gasField[0])<<2 | uint16(gasField[1])>>6
	gasRange := gasField[1] & 0x0F
	heatStable := gasField[1]&0x10 != 0

	temp := math.Round(tempC*10) / 10
	hum := math.Round(humidity*10) / 10

	data := connector.NewAirQualityData()
	data.Temperature = &temp
	data.Humidity = &hum
	data.Location = c.cfg.Location
	data.Source = "bme680"

	// IAQ score 0-500 from gas resistance. Mirrors Python:
	// ratio = gas_resistance / gas_baseline
	// aqi = max(0, min(500, round(25 / max(ratio, 0.01))))
	if heatStable {
		gasResistance := c.calib.compensateGas(gasADC, gasRange)
		ratio := gasResistance / c.cfg.GasBaseline
		if ratio < 0.01 {
			ratio = 0.01
		}
		aqi := int(math.Round(25 / ratio))
		if aqi > 500 {
			aqi = 500
		}
		if aqi < 0 {
			aqi = 0
		}
		data.AQI = &aqi
	}

	return data, nil
}

// --- Low-level I2C helpers -------------------------------------------------

func ioctlSetSlave(f *os.File, addr uint16) error {
	_, _, errno := unix.Syscall(unix.SYS_IOCTL, f.Fd(), i2cSlaveIoctl, uintptr(addr))
	if errno != 0 {
		return fmt.Errorf("ioctl I2C_SLAVE: %w", errno)
	}
	return nil
}

func writeReg(f *os.File, reg, value byte) error {
	_, err := f.Write([]byte{reg, value})
	return err
}

func readReg(f *os.File, reg byte) (byte, error) {
	buf, err := readBlock(f, reg, 1)
	if err != nil {
		return 0, err
	}
	return buf[0], nil
}

func readBlock(f *os.File, reg byte, length int) ([]byte, error) {
	if _, err := f.Write([]byte{reg}); err != nil {
		return nil, err
	}
	buf := make([]byte, length)
	if _, err := f.Read(buf); err != nil {
		return nil, err
	}
	return buf, nil
}
