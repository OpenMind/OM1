package pms5003

import (
	"bytes"
	"context"
	"log"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"go.bug.st/serial"
)

// mockPort implements serial.Port over an in-memory byte buffer, so tests
// don't need real hardware.
type mockPort struct {
	buf       *bytes.Reader
	closeErr  error
	closed    bool
	readErr   error
	readCalls int
}

func newMockPort(data []byte) *mockPort {
	return &mockPort{buf: bytes.NewReader(data)}
}

func (m *mockPort) SetMode(mode *serial.Mode) error { return nil }

func (m *mockPort) Read(p []byte) (int, error) {
	m.readCalls++
	if m.readErr != nil {
		return 0, m.readErr
	}
	n, err := m.buf.Read(p)
	if err != nil { // io.EOF at end of buffer -> mimic timeout (0 bytes, no error)
		return 0, nil
	}
	return n, nil
}

func (m *mockPort) Write(p []byte) (int, error)                          { return len(p), nil }
func (m *mockPort) Drain() error                                         { return nil }
func (m *mockPort) ResetInputBuffer() error                              { return nil }
func (m *mockPort) ResetOutputBuffer() error                             { return nil }
func (m *mockPort) SetDTR(dtr bool) error                                { return nil }
func (m *mockPort) SetRTS(rts bool) error                                { return nil }
func (m *mockPort) GetModemStatusBits() (*serial.ModemStatusBits, error) { return &serial.ModemStatusBits{}, nil }
func (m *mockPort) SetReadTimeout(t time.Duration) error                 { return nil }
func (m *mockPort) Break(d time.Duration) error                          { return nil }
func (m *mockPort) Close() error {
	m.closed = true
	return m.closeErr
}

// buildValidFrame constructs a valid 32-byte PMS5003 frame for the given
// PM2.5 and PM10 values (offsets [6:8] and [8:10]), with a correct checksum.
func buildValidFrame(pm25, pm10 int) []byte {
	frame := make([]byte, 32)
	frame[0] = 0x42
	frame[1] = 0x4D
	frame[6] = byte(pm25 >> 8)
	frame[7] = byte(pm25)
	frame[8] = byte(pm10 >> 8)
	frame[9] = byte(pm10)

	var checksum uint32
	for _, b := range frame[:30] {
		checksum += uint32(b)
	}
	checksum &= 0xFFFF
	frame[30] = byte(checksum >> 8)
	frame[31] = byte(checksum)
	return frame
}

func newTestConnector(port *mockPort) *Connector {
	c := New(Config{Location: "TestLab"}, log.Default())
	c.port = port
	return c
}

func TestRead_ValidFrame(t *testing.T) {
	frame := buildValidFrame(35, 50)
	c := newTestConnector(newMockPort(frame))

	data, err := c.Read(context.Background())
	require.NoError(t, err)
	require.NotNil(t, data)
	require.Equal(t, 35.0, *data.PM25)
	require.Equal(t, 50.0, *data.PM10)
	require.Equal(t, "TestLab", data.Location)
	require.Equal(t, "pms5003", data.Source)
	require.NotNil(t, data.AQI)
}

func TestRead_NotConnected(t *testing.T) {
	c := New(Config{}, log.Default())
	data, err := c.Read(context.Background())
	require.NoError(t, err)
	require.Nil(t, data)
}

func TestRead_BadChecksum(t *testing.T) {
	frame := buildValidFrame(35, 50)
	frame[31] ^= 0xFF // corrupt checksum
	c := newTestConnector(newMockPort(frame))

	data, err := c.Read(context.Background())
	require.NoError(t, err)
	require.Nil(t, data)
}

func TestRead_NoHeaderFound(t *testing.T) {
	// 64+ junk bytes, no 0x42 0x4D anywhere -> readFrame loop exits via timeout (0 bytes).
	junk := bytes.Repeat([]byte{0xFF}, 70)
	c := newTestConnector(newMockPort(junk))

	data, err := c.Read(context.Background())
	require.NoError(t, err)
	require.Nil(t, data)
}

func TestRead_IncompleteFrame(t *testing.T) {
	frame := buildValidFrame(35, 50)
	truncated := frame[:20] // header ok, but frame cut short
	c := newTestConnector(newMockPort(truncated))

	data, err := c.Read(context.Background())
	require.NoError(t, err)
	require.Nil(t, data)
}

func TestDisconnect_ClosesPort(t *testing.T) {
	port := newMockPort(nil)
	c := newTestConnector(port)

	err := c.Disconnect(context.Background())
	require.NoError(t, err)
	require.True(t, port.closed)
}

func TestDisconnect_NilPort(t *testing.T) {
	c := New(Config{}, log.Default())
	err := c.Disconnect(context.Background())
	require.NoError(t, err)
}

func TestName(t *testing.T) {
	c := New(Config{}, log.Default())
	require.Equal(t, "pms5003", c.Name())
}

func TestNew_Defaults(t *testing.T) {
	c := New(Config{}, log.Default())
	require.Equal(t, "/dev/ttyUSB0", c.cfg.Port)
	require.Equal(t, "Robot", c.cfg.Location)
}

func TestPM25ToAQI_Breakpoints(t *testing.T) {
	require.Equal(t, 0, pm25ToAQI(0))
	require.Equal(t, 50, pm25ToAQI(12.0))
	require.InDelta(t, 100, pm25ToAQI(35.4), 1)
	require.Equal(t, 500, pm25ToAQI(1000)) // above scale, capped
}
