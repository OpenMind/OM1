package go2

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/geometry"
	"github.com/openmind/om1/internal/httpclient"
	"github.com/openmind/om1/internal/logger"
	"github.com/openmind/om1/internal/util"
)

const (
	defaultLocationsURL     = "http://localhost:5000/maps/locations/list"
	defaultLocationsTimeout = 5 * time.Second
	defaultLocationsRefresh = 30 * time.Second
)

type Location struct {
	Name string        `json:"name"`
	Pose geometry.Pose `json:"pose"`
}

type LocationsProvider struct {
	log             *zap.Logger
	baseURL         string
	apiKey          string
	timeout         time.Duration
	refreshInterval time.Duration
	client          *http.Client

	mu        sync.RWMutex
	locations map[string]Location

	stopOnce sync.Once
	stop     chan struct{}
	done     chan struct{}
}

// NewLocationsProvider constructs a LocationsProvider with the given configuration.
func NewLocationsProvider(baseURL, apiKey string, timeout, refreshInterval time.Duration) *LocationsProvider {
	if baseURL == "" {
		baseURL = defaultLocationsURL
	}
	if timeout <= 0 {
		timeout = defaultLocationsTimeout
	}
	if refreshInterval <= 0 {
		refreshInterval = defaultLocationsRefresh
	}

	return &LocationsProvider{
		log:             logger.Get().Named("unitree_go2_locations"),
		baseURL:         baseURL,
		apiKey:          apiKey,
		timeout:         timeout,
		refreshInterval: refreshInterval,
		client:          httpclient.Default(),
		locations:       map[string]Location{},
		stop:            make(chan struct{}),
		done:            make(chan struct{}),
	}
}

// Start begins the background loop that periodically refreshes the location cache until Stop is called.
func (p *LocationsProvider) Start() {
	go func() {
		defer close(p.done)

		p.fetch()

		ticker := time.NewTicker(p.refreshInterval)
		defer ticker.Stop()

		for {
			select {
			case <-p.stop:
				return
			case <-ticker.C:
				p.fetch()
			}
		}
	}()
}

// Stop terminates the background refresh loop.
func (p *LocationsProvider) Stop() {
	p.stopOnce.Do(func() {
		close(p.stop)
		<-p.done
	})
}

// fetch retrieves the location list from the API and updates the cache.
func (p *LocationsProvider) fetch() {
	if p.baseURL == "" {
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), p.timeout)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, p.baseURL, nil)
	if err != nil {
		p.log.Error("build request failed", zap.Error(err))
		return
	}
	req.Header.Set("x-api-key", p.apiKey)

	resp, err := p.client.Do(req)
	if err != nil {
		p.log.Error("location list request failed", zap.Error(err))
		return
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		p.log.Error("location list API returned error", zap.Int("status", resp.StatusCode))
		return
	}

	locations, err := parseLocations(resp.Body)
	if err != nil {
		p.log.Error("failed to parse location list", zap.Error(err))
		return
	}

	p.mu.Lock()
	p.locations = locations
	p.mu.Unlock()
}

// parseLocation handles both map and list formats for the location API response, with an optional "message" envelope.
func parseLocations(r io.Reader) (map[string]Location, error) {
	var raw json.RawMessage
	if err := json.NewDecoder(r).Decode(&raw); err != nil {
		return nil, err
	}

	var envelope struct {
		Message *string `json:"message"`
	}
	if err := json.Unmarshal(raw, &envelope); err == nil && envelope.Message != nil {
		raw = json.RawMessage(*envelope.Message)
	}

	parsed := map[string]Location{}

	var asMap map[string]Location
	if err := json.Unmarshal(raw, &asMap); err == nil {
		for key, loc := range asMap {
			if loc.Name == "" {
				loc.Name = key
			}
			parsed[util.TrimLower(key)] = loc
		}
		return parsed, nil
	}

	var asList []Location
	if err := json.Unmarshal(raw, &asList); err != nil {
		return nil, err
	}
	for _, loc := range asList {
		name := strings.TrimSpace(loc.Name)
		if name == "" {
			continue
		}
		parsed[util.TrimLower(name)] = loc
	}
	return parsed, nil
}

// GetLocation looks up a location by label, returning it and whether it was found.
func (p *LocationsProvider) GetLocation(label string) (Location, bool) {
	key := util.TrimLower(label)
	if key == "" {
		return Location{}, false
	}
	p.mu.RLock()
	defer p.mu.RUnlock()
	loc, ok := p.locations[key]
	return loc, ok
}

// AllNames returns the display names of all cached locations.
func (p *LocationsProvider) AllNames() []string {
	p.mu.RLock()
	defer p.mu.RUnlock()
	names := make([]string, 0, len(p.locations))
	for key, loc := range p.locations {
		if loc.Name != "" {
			names = append(names, loc.Name)
		} else {
			names = append(names, key)
		}
	}
	return names
}
