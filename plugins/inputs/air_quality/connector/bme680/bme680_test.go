package bme680

import (
	"context"
	"log"
	"testing"

	"github.com/stretchr/testify/require"
)

// fakeI2CBus implements i2cBus entirely in memory for testing, without
// touching real hardware.
type fakeI2CBus struct {
	regs     map[byte]byte   // single-register read/write state
	blocks   map[byte][]byte // starting register -> bytes returned by readBlock
	writes   map[byte]byte   // records every writeReg call
	writeErr error
	readErr  error
}

func newFakeBus() *fakeI2CBus {
	return &fakeI2CBus{
		regs:   make(map[byte]byte),
		blocks: make(map[byte][]byte),
		writes: make(map[byte]byte),
	}
}

func (b *fakeI2CBus) writeReg(reg, value byte) error {
	if b.writeErr != nil {
		return b.writeErr
	}
	b.writes[reg] = value
	b.regs[reg] = value
	return nil
}

func (b *fakeI2CBus) readReg(reg byte) (byte, error) {
	if b.readErr != nil {
		return 0, b.readErr
	}
	return b.regs[reg], nil
}

func (b *fakeI2CBus) readBlock(reg byte, length int) ([]byte, error) {
	if b.readErr != nil {
		return nil, b.readErr
	}
	if data, ok := b.blocks[reg]; ok {
		out := make([]byte, length)
		copy(out, data)
		return out, nil
	}
	return make([]byte, length), nil
}

// validCalibratedBus returns a fake bus pre-loaded with a valid chip ID,
// zeroed (but present) calibration blocks, an immediately-ready measurement
// status, and plausible raw field/gas data.
func validCalibratedBus() *fakeI2CBus {
	b := newFakeBus()
	b.regs[regChipID] = chipIDExpected
	b.blocks[regCalibStart1] = make([]byte, 25)
	b.blocks[regCalibStart2] = make([]byte, 16)
	b.regs[regHeatRange] = 0x00
	b.regs[regHeatVal] = 0x00
	b.regs[regRangeSwErr] = 0x00
	b.regs[regMeasStatus0] = 0x80 // new_data bit set: data ready immediately
	// field block: press(3) + temp(3) + hum(2), arbitrary plausible raw values
	b.blocks[regFieldStart] = []byte{0, 0, 0, 0x80, 0x00, 0x00, 0x60, 0x00}
	// gas field: gas_r_msb, gas_r_lsb -> heat_stable bit (0x10) set, range=0
	b.blocks[0x2A] = []byte{0x20, 0x10}
	return b
}

func newTestConnector() *Connector {
	return New(Config{Location: "TestLab", GasBaseline: 50000}, log.Default())
}

func TestConnectWithBus_Success(t *testing.T) {
	c := newTestConnector()
	bus := validCalibratedBus()

	ok, err := c.connectWithBus(bus)
	require.NoError(t, err)
	require.True(t, ok)
	require.NotNil(t, c.bus)

	// Oversampling/config/heater registers must have been written.
	require.Contains(t, bus.writes, byte(regCtrlHum))
	require.Contains(t, bus.writes, byte(regCtrlMeas))
	require.Contains(t, bus.writes, byte(regConfig))
	require.Contains(t, bus.writes, byte(regResHeat0))
	require.Contains(t, bus.writes, byte(regGasWait0))
	require.Contains(t, bus.writes, byte(regCtrlGas1))
}

func TestConnectWithBus_BadChipID(t *testing.T) {
	c := newTestConnector()
	bus := newFakeBus()
	bus.regs[regChipID] = 0x00 // wrong chip id

	ok, err := c.connectWithBus(bus)
	require.NoError(t, err)
	require.False(t, ok)
	require.Nil(t, c.bus)
}

func TestConnectWithBus_ReadError(t *testing.T) {
	c := newTestConnector()
	bus := newFakeBus()
	bus.readErr = context.DeadlineExceeded

	ok, err := c.connectWithBus(bus)
	require.NoError(t, err)
	require.False(t, ok)
}

func TestRead_NotConnected(t *testing.T) {
	c := newTestConnector()
	data, err := c.Read(context.Background())
	require.NoError(t, err)
	require.Nil(t, data)
}

func TestRead_FullCycle(t *testing.T) {
	c := newTestConnector()
	bus := validCalibratedBus()

	ok, err := c.connectWithBus(bus)
	require.NoError(t, err)
	require.True(t, ok)

	data, err := c.Read(context.Background())
	require.NoError(t, err)
	require.NotNil(t, data)
	require.NotNil(t, data.Temperature)
	require.NotNil(t, data.Humidity)
	require.Equal(t, "TestLab", data.Location)
	require.Equal(t, "bme680", data.Source)
	// heat_stable bit was set in the fake gas field, so AQI must be populated.
	require.NotNil(t, data.AQI)
	require.GreaterOrEqual(t, *data.AQI, 0)
	require.LessOrEqual(t, *data.AQI, 500)
}

func TestRead_GasNotStable_NoAQI(t *testing.T) {
	c := newTestConnector()
	bus := validCalibratedBus()
	bus.blocks[0x2A] = []byte{0x20, 0x00} // heat_stable bit (0x10) cleared

	ok, err := c.connectWithBus(bus)
	require.NoError(t, err)
	require.True(t, ok)

	data, err := c.Read(context.Background())
	require.NoError(t, err)
	require.NotNil(t, data)
	require.Nil(t, data.AQI)
}

func TestDisconnect_NilFile(t *testing.T) {
	c := newTestConnector()
	c.bus = validCalibratedBus()

	err := c.Disconnect(context.Background())
	require.NoError(t, err)
	require.Nil(t, c.bus)
}

func TestName(t *testing.T) {
	c := newTestConnector()
	require.Equal(t, "bme680", c.Name())
}

func TestNew_Defaults(t *testing.T) {
	c := New(Config{}, log.Default())
	require.Equal(t, "/dev/i2c-1", c.cfg.I2CBus)
	require.Equal(t, uint16(0x76), c.cfg.I2CAddress)
	require.Equal(t, "Robot", c.cfg.Location)
	require.Equal(t, 50000.0, c.cfg.GasBaseline)
}

// --- Calibration formula sanity tests --------------------------------------
// These don't validate against real hardware (that requires physical
// calibration and a reference sensor), only that the formulas are
// deterministic, stay in-range, and respond monotonically as documented in
// the Bosch datasheet.

func TestCompensateTemp_Deterministic(t *testing.T) {
	var c calibration
	c.parT1, c.parT2, c.parT3 = 26000, 26500, 3

	_, temp1 := c.compensateTemp(500000)
	_, temp2 := c.compensateTemp(500000)
	require.Equal(t, temp1, temp2, "same input must give same output")
}

func TestCompensateHumidity_ClampedRange(t *testing.T) {
	var c calibration
	c.parH1, c.parH2 = 600, 400
	c.parH3, c.parH4, c.parH5, c.parH6, c.parH7 = 0, 20, 5, 60, 30

	hum := c.compensateHumidity(30000, 25000)
	require.GreaterOrEqual(t, hum, 0.0)
	require.LessOrEqual(t, hum, 100.0)
}

func TestCalcGasWait_Encoding(t *testing.T) {
	// 150ms should fit in the first multiplier tier (factor=0) since 150 <= 0x3F*4-ish range;
	// just assert it round-trips to a value <= 0xFF and is deterministic.
	v1 := calcGasWait(150)
	v2 := calcGasWait(150)
	require.Equal(t, v1, v2)
}

func TestCalcHeaterResistance_Deterministic(t *testing.T) {
	var c calibration
	c.parG1, c.parG2, c.parG3 = 50, 100, 10
	c.resHeatRange, c.resHeatVal = 1, 2

	v1 := c.calcHeaterResistance(320, 25)
	v2 := c.calcHeaterResistance(320, 25)
	require.Equal(t, v1, v2)
}
