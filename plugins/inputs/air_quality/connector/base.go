package connector

import (
	"context"
	"math"
)

// AirQualityData is the standardized data structure returned by all connectors.
// Mirrors the Python dataclass AirQualityData in connector/base.py.
type AirQualityData struct {
	AQI         *int
	PM25        *float64
	PM10        *float64
	CO          *float64
	NO2         *float64
	SO2         *float64
	O3          *float64
	Temperature *float64
	Humidity    *float64
	Location    string
	Source      string
}

// NewAirQualityData returns a data struct with Python-equivalent defaults
// (location="Unknown", source="Unknown").
func NewAirQualityData() *AirQualityData {
	return &AirQualityData{
		Location: "Unknown",
		Source:   "Unknown",
	}
}

type aqiLevel struct {
	threshold   float64
	label       string
	description string
}

// AQILevels mirrors the Python AQI_LEVELS constant (US EPA AQI scale).
var AQILevels = []aqiLevel{
	{50, "GOOD", "Air quality is satisfactory."},
	{100, "MODERATE", "Acceptable; some pollutants may concern sensitive groups."},
	{150, "UNHEALTHY FOR SENSITIVE GROUPS", "Sensitive groups may experience health effects."},
	{200, "UNHEALTHY", "Everyone may begin to experience health effects."},
	{300, "VERY UNHEALTHY", "Health alert: everyone may experience serious effects."},
	{math.Inf(1), "HAZARDOUS", "Health warning: emergency conditions for entire population."},
}

// GetAQILevel mirrors Python's get_aqi_level(aqi) -> (label, description).
func GetAQILevel(aqi int) (label string, description string) {
	for _, lvl := range AQILevels {
		if float64(aqi) <= lvl.threshold {
			return lvl.label, lvl.description
		}
	}
	last := AQILevels[len(AQILevels)-1]
	return last.label, last.description
}

// AirQualityConnector is the interface every connector must implement.
// Mirrors the Python ABC AirQualityConnector.
//
// Connect returns (bool, error): bool mirrors Python's True/False success signal
// (e.g. missing API key, sensor not found), error is for unexpected failures.
// Callers should treat `!ok` the same as Python treating `connected == False`.
type AirQualityConnector interface {
	Connect(ctx context.Context) (ok bool, err error)
	Read(ctx context.Context) (*AirQualityData, error)
	Disconnect(ctx context.Context) error
	Name() string
}
