package aqicn

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/openmind/om1/plugins/inputs/air_quality/connector"
)

// Config mirrors the `config: dict` passed into Python's AqicnConnector.__init__.
type Config struct {
	APIKey    string  // config["api_key"]
	Latitude  float64 // config["latitude"], default -6.2088
	Longitude float64 // config["longitude"], default 106.8456
}

// Connector implements connector.AirQualityConnector for the AQICN cloud API.
// Mirrors Python's AqicnConnector.
type Connector struct {
	cfg        Config
	httpClient *http.Client
	logger     *log.Logger
}

// New creates a new AQICN connector, applying the same defaults as the Python version.
func New(cfg Config, logger *log.Logger) *Connector {
	if cfg.Latitude == 0 && cfg.Longitude == 0 {
		cfg.Latitude = -6.2088
		cfg.Longitude = 106.8456
	}
	if logger == nil {
		logger = log.Default()
	}
	return &Connector{
		cfg:        cfg,
		httpClient: &http.Client{Timeout: 10 * time.Second},
		logger:     logger,
	}
}

func (c *Connector) Name() string {
	return "aqicn"
}

// Connect validates the API key and confirms the connector is ready.
// Mirrors Python's `connect()`.
func (c *Connector) Connect(ctx context.Context) (bool, error) {
	if c.cfg.APIKey == "" {
		c.logger.Println("AqicnConnector: no API key provided")
		return false, nil
	}
	return true, nil
}

// Disconnect is a no-op: stateless HTTP connector requires no teardown.
func (c *Connector) Disconnect(ctx context.Context) error {
	return nil
}

type iaqiEntry struct {
	V *float64 `json:"v"`
}

type aqicnPayload struct {
	Status string `json:"status"`
	Data   struct {
		AQI  json.Number `json:"aqi"`
		City struct {
			Name string `json:"name"`
		} `json:"city"`
		IAQI struct {
			PM25 *iaqiEntry `json:"pm25"`
			PM10 *iaqiEntry `json:"pm10"`
			CO   *iaqiEntry `json:"co"`
			NO2  *iaqiEntry `json:"no2"`
			SO2  *iaqiEntry `json:"so2"`
			O3   *iaqiEntry `json:"o3"`
			T    *iaqiEntry `json:"t"`
			H    *iaqiEntry `json:"h"`
		} `json:"iaqi"`
	} `json:"data"`
}

// Read fetches air quality data from the AQICN API.
// Returns (nil, nil) on any handled failure, mirroring Python's `return None`.
func (c *Connector) Read(ctx context.Context) (*connector.AirQualityData, error) {
	if c.cfg.APIKey == "" {
		return nil, nil
	}

	url := fmt.Sprintf("https://api.waqi.info/feed/geo:%v;%v/?token=%s", c.cfg.Latitude, c.cfg.Longitude, c.cfg.APIKey)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		c.logger.Printf("AqicnConnector: unexpected error: %v", err)
		return nil, nil
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		c.logger.Printf("AqicnConnector: network error: %v", err)
		return nil, nil
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		c.logger.Printf("AqicnConnector: HTTP %d", resp.StatusCode)
		return nil, nil
	}

	var payload aqicnPayload
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		c.logger.Printf("AqicnConnector: unexpected error: %v", err)
		return nil, nil
	}

	if payload.Status != "ok" {
		c.logger.Printf("AqicnConnector: API error, status=%q", payload.Status)
		return nil, nil
	}

	return c.parse(payload), nil
}

// parse mirrors Python's `_parse(payload)`.
func (c *Connector) parse(payload aqicnPayload) *connector.AirQualityData {
	data := connector.NewAirQualityData()

	// aqi_raw = data.get("aqi", "-"); aqi = int(aqi_raw) if aqi_raw not in ("-", None) else None
	if aqiFloat, err := payload.Data.AQI.Float64(); err == nil {
		aqi := int(aqiFloat)
		data.AQI = &aqi
	}

	get := func(e *iaqiEntry) *float64 {
		if e == nil {
			return nil
		}
		return e.V
	}

	data.PM25 = get(payload.Data.IAQI.PM25)
	data.PM10 = get(payload.Data.IAQI.PM10)
	data.CO = get(payload.Data.IAQI.CO)
	data.NO2 = get(payload.Data.IAQI.NO2)
	data.SO2 = get(payload.Data.IAQI.SO2)
	data.O3 = get(payload.Data.IAQI.O3)
	data.Temperature = get(payload.Data.IAQI.T)
	data.Humidity = get(payload.Data.IAQI.H)

	if payload.Data.City.Name != "" {
		data.Location = payload.Data.City.Name
	} else {
		data.Location = "Unknown"
	}
	data.Source = "aqicn"

	return data
}
