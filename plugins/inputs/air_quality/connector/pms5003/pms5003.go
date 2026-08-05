package pms5003

import (
	"context"
	"log"
	"time"

	"go.bug.st/serial"

	"github.com/openmind/om1/plugins/inputs/air_quality/connector"
)

const (
	baudRate    = 9600
	frameLength = 32
)

var frameStart = [2]byte{0x42, 0x4D}

// Config mirrors the `config: dict` passed into Python's PMS5003Connector.__init__.
type Config struct {
	Port     string // config["port"], default "/dev/ttyUSB0"
	Location string // config["location"], default "Robot"
}

// Connector implements connector.AirQualityConnector for the PMS5003/PMS7003 sensor.
// Mirrors Python's PMS5003Connector.
type Connector struct {
	cfg    Config
	port   serial.Port
	logger *log.Logger
}

// New creates a new PMS5003 connector, applying the same defaults as the Python version.
func New(cfg Config, logger *log.Logger) *Connector {
	if cfg.Port == "" {
		cfg.Port = "/dev/ttyUSB0"
	}
	if cfg.Location == "" {
		cfg.Location = "Robot"
	}
	if logger == nil {
		logger = log.Default()
	}
	return &Connector{cfg: cfg, logger: logger}
}

func (c *Connector) Name() string {
	return "pms5003"
}

// Connect opens the serial port. Mirrors Python's `connect()`.
func (c *Connector) Connect(ctx context.Context) (bool, error) {
	mode := &serial.Mode{
		BaudRate: baudRate,
		DataBits: 8,
		Parity:   serial.NoParity,
		StopBits: serial.OneStopBit,
	}

	port, err := serial.Open(c.cfg.Port, mode)
	if err != nil {
		c.logger.Printf("PMS5003Connector: failed to connect: %v", err)
		return false, nil
	}
	port.SetReadTimeout(2 * time.Second)

	c.port = port
	c.logger.Printf("PMS5003Connector: connected on %s", c.cfg.Port)
	return true, nil
}

// Disconnect closes the serial port if open. Mirrors Python's `disconnect()`.
func (c *Connector) Disconnect(ctx context.Context) error {
	if c.port != nil {
		if err := c.port.Close(); err != nil {
			return err
		}
		c.logger.Println("PMS5003Connector: disconnected")
	}
	return nil
}

// Read reads one frame from the sensor. Runs in a goroutine, mirroring Python's
// `loop.run_in_executor` usage to avoid blocking the caller.
func (c *Connector) Read(ctx context.Context) (*connector.AirQualityData, error) {
	if c.port == nil {
		c.logger.Println("PMS5003Connector: not connected")
		return nil, nil
	}

	type result struct {
		data *connector.AirQualityData
		err  error
	}
	resCh := make(chan result, 1)

	go func() {
		frame, err := c.readFrame()
		if err != nil {
			c.logger.Printf("PMS5003Connector: read error: %v", err)
			resCh <- result{nil, nil}
			return
		}
		if frame == nil {
			resCh <- result{nil, nil}
			return
		}
		resCh <- result{c.parse(frame), nil}
	}()

	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case res := <-resCh:
		return res.data, res.err
	}
}

// readFrame reads and validates one 32-byte frame (blocking).
// Mirrors Python's `_read_frame()`.
func (c *Connector) readFrame() ([]byte, error) {
	one := make([]byte, 1)

	// Sync to frame start bytes 0x42 0x4D
	for {
		n, err := c.port.Read(one)
		if err != nil {
			return nil, err
		}
		if n == 0 {
			return nil, nil // timeout, mirrors Python's "if not byte: return None"
		}
		if one[0] == frameStart[0] {
			n2, err := c.port.Read(one)
			if err != nil {
				return nil, err
			}
			if n2 > 0 && one[0] == frameStart[1] {
				break
			}
		}
	}

	rest := make([]byte, frameLength-2)
	total := 0
	for total < len(rest) {
		n, err := c.port.Read(rest[total:])
		if err != nil {
			return nil, err
		}
		if n == 0 {
			break
		}
		total += n
	}
	if total != len(rest) {
		return nil, nil // incomplete frame, mirrors Python's length check -> None
	}

	frame := append([]byte{frameStart[0], frameStart[1]}, rest...)

	// Validate checksum: sum of bytes 0..29 == bytes 30..31 (big-endian)
	var checksum uint32
	for _, b := range frame[:30] {
		checksum += uint32(b)
	}
	checksum &= 0xFFFF
	expected := uint32(frame[30])<<8 | uint32(frame[31])
	if checksum != expected {
		c.logger.Println("PMS5003Connector: checksum mismatch")
		return nil, nil
	}

	return frame, nil
}

// parse parses a validated 32-byte frame. Mirrors Python's `_parse(frame)`.
// Frame layout (big-endian): [6:8] PM2.5 standard, [8:10] PM10 standard.
func (c *Connector) parse(frame []byte) *connector.AirQualityData {
	word := func(offset int) int {
		return int(frame[offset])<<8 | int(frame[offset+1])
	}

	pm25 := float64(word(6))
	pm10 := float64(word(8))
	aqi := pm25ToAQI(pm25)

	data := connector.NewAirQualityData()
	data.AQI = &aqi
	data.PM25 = &pm25
	data.PM10 = &pm10
	data.Location = c.cfg.Location
	data.Source = "pms5003"
	return data
}

// pm25ToAQI mirrors Python's `_pm25_to_aqi(pm25)`: US EPA breakpoint formula.
func pm25ToAQI(pm25 float64) int {
	type bp struct{ cLow, cHigh, aqiLow, aqiHigh float64 }
	breakpoints := []bp{
		{0.0, 12.0, 0, 50},
		{12.1, 35.4, 51, 100},
		{35.5, 55.4, 101, 150},
		{55.5, 150.4, 151, 200},
		{150.5, 250.4, 201, 300},
		{250.5, 350.4, 301, 400},
		{350.5, 500.4, 401, 500},
	}
	for _, b := range breakpoints {
		if pm25 >= b.cLow && pm25 <= b.cHigh {
			aqi := ((b.aqiHigh-b.aqiLow)/(b.cHigh-b.cLow))*(pm25-b.cLow) + b.aqiLow
			return int(aqi + 0.5) // round half up, mirrors Python's round()
		}
	}
	return 500
}

