package aqicn

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestConnect_NoAPIKey(t *testing.T) {
	c := New(Config{}, nil)
	ok, err := c.Connect(context.Background())
	require.NoError(t, err)
	require.False(t, ok)
}

func TestConnect_WithAPIKey(t *testing.T) {
	c := New(Config{APIKey: "token"}, nil)
	ok, err := c.Connect(context.Background())
	require.NoError(t, err)
	require.True(t, ok)
}

func TestRead_Success(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{
			"status": "ok",
			"data": {
				"aqi": 99,
				"city": {"name": "Semarang, Indonesia"},
				"iaqi": {
					"pm25": {"v": 99},
					"t": {"v": 32.9},
					"h": {"v": 53.1}
				}
			}
		}`))
	}))
	defer server.Close()

	c := New(Config{APIKey: "token", BaseURL: server.URL}, nil)
	data, err := c.Read(context.Background())

	require.NoError(t, err)
	require.NotNil(t, data)
	require.Equal(t, 99, *data.AQI)
	require.Equal(t, 99.0, *data.PM25)
	require.Equal(t, 32.9, *data.Temperature)
	require.Equal(t, 53.1, *data.Humidity)
	require.Equal(t, "Semarang, Indonesia", data.Location)
	require.Equal(t, "aqicn", data.Source)
}

func TestRead_NoAPIKey(t *testing.T) {
	c := New(Config{}, nil)
	data, err := c.Read(context.Background())
	require.NoError(t, err)
	require.Nil(t, data)
}

func TestRead_HTTPError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer server.Close()

	c := New(Config{APIKey: "token", BaseURL: server.URL}, nil)
	data, err := c.Read(context.Background())
	require.NoError(t, err)
	require.Nil(t, data)
}

func TestRead_APIStatusError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"status": "error", "data": "Invalid key"}`))
	}))
	defer server.Close()

	c := New(Config{APIKey: "token", BaseURL: server.URL}, nil)
	data, err := c.Read(context.Background())
	require.NoError(t, err)
	require.Nil(t, data)
}

func TestRead_MalformedJSON(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`not json`))
	}))
	defer server.Close()

	c := New(Config{APIKey: "token", BaseURL: server.URL}, nil)
	data, err := c.Read(context.Background())
	require.NoError(t, err)
	require.Nil(t, data)
}

func TestRead_MissingPollutants(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"status": "ok", "data": {"aqi": "-", "city": {}, "iaqi": {}}}`))
	}))
	defer server.Close()

	c := New(Config{APIKey: "token", BaseURL: server.URL}, nil)
	data, err := c.Read(context.Background())
	require.NoError(t, err)
	require.NotNil(t, data)
	require.Nil(t, data.AQI)
	require.Nil(t, data.PM25)
	require.Equal(t, "Unknown", data.Location)
}

func TestDisconnect_NoOp(t *testing.T) {
	c := New(Config{}, nil)
	err := c.Disconnect(context.Background())
	require.NoError(t, err)
}

func TestName(t *testing.T) {
	c := New(Config{}, nil)
	require.Equal(t, "aqicn", c.Name())
}

func TestNew_DefaultLocation(t *testing.T) {
	c := New(Config{}, nil)
	require.Equal(t, -6.2088, c.cfg.Latitude)
	require.Equal(t, 106.8456, c.cfg.Longitude)
}
