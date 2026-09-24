package httpclient

import (
	"net/http"
	"sync"
	"time"
)

var (
	once          sync.Once
	defaultClient *http.Client
)

// Default returns a process-wide HTTP client shared by all plugins.
// The transport is tuned for concurrent requests to a small number of
// external API hosts and forces HTTP/2 negotiation on TLS connections.
func Default() *http.Client {
	once.Do(func() {
		transport := &http.Transport{
			MaxIdleConns:        500,
			MaxIdleConnsPerHost: 100,
			MaxConnsPerHost:     0,
			IdleConnTimeout:     90 * time.Second,
			DisableCompression:  false,
			ForceAttemptHTTP2:   true,
		}
		defaultClient = &http.Client{Transport: transport, Timeout: 60 * time.Second}
	})
	return defaultClient
}
