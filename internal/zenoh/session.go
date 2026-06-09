package zenoh

import (
	"sync"

	"github.com/openmind/om1/internal/cloudsession"
)

type Session interface {
	// DeclarePublisher declares a publisher for the given key and returns it.
	DeclarePublisher(key string) (Publisher, error)

	// DeclareSubscriber declares a subscriber for the given key and handler function.
	DeclareSubscriber(key string, handler func([]byte)) (Subscriber, error)

	//	Put publishes data to the given key (one-shot).
	Put(key string, data []byte) error

	// Close properly closes the session and releases resources.
	Close()
}

type Publisher interface {
	// Put publishes data on the publisher's key expression.
	Put(data []byte) error

	// Drop undeclares the publisher and releases resources.
	Drop()
}

type Subscriber interface {
	// Drop undeclares the subscriber and releases resources.
	Drop()
}

// Options configures how a Session is opened.
type Options struct {
	// Endpoint is the local Zenoh router endpoint (e.g. tcp/127.0.0.1:7447).
	Endpoint string

	// LocalNetwork forces the session to connect to a local Zenoh router and not attempt discovery.
	LocalNetwork bool

	// UseSim enables the hybrid session that routes cloud topics to a cloud session and non-cloud topics to a local Zenoh session.
	UseSim bool

	// APIKey is the API key for authenticating with the cloud broker.
	APIKey string

	// CloudURL is the WebSocket URL of the cloud broker.
	CloudURL string
}

var (
	defaultOptsMu sync.RWMutex
	defaultOpts   Options
)

func SetDefaultOptions(opts Options) {
	if opts.CloudURL == "" {
		opts.CloudURL = cloudsession.DefaultURL
	}
	defaultOptsMu.Lock()
	defaultOpts = opts
	defaultOptsMu.Unlock()
}

func defaultOptions() Options {
	defaultOptsMu.RLock()
	defer defaultOptsMu.RUnlock()
	return defaultOpts
}

// Open creates a new Session with the given endpoint or the default if empty.
func Open(endpoint ...string) (Session, error) {
	opts := defaultOptions()
	opts.LocalNetwork = true
	if len(endpoint) > 0 {
		opts.Endpoint = endpoint[0]
	}

	return OpenWithOptions(opts)
}

// OpenWithOptions creates a Session with explicit options.
func OpenWithOptions(opts Options) (Session, error) {
	if opts.UseSim {
		return newHybridSession(opts), nil
	}

	return openZenoh(opts)
}
