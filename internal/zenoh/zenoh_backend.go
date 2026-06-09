package zenoh

import (
	"fmt"

	"github.com/eclipse-zenoh/zenoh-go/zenoh"
	"go.uber.org/zap"

	"github.com/openmind/om1/internal/logger"
)

const defaultEndpoint = "tcp/127.0.0.1:7447"

type zenohSession struct {
	session zenoh.Session
}

type zenohPublisher struct {
	publisher zenoh.Publisher
}

type zenohSubscriber struct {
	subscriber zenoh.Subscriber
}

func openZenoh(opts Options) (*zenohSession, error) {
	endpoint := opts.Endpoint
	if endpoint == "" {
		endpoint = defaultEndpoint
	}

	if opts.LocalNetwork {
		session, err := openClient(endpoint)
		if err == nil {
			return session, nil
		}

		logger.Get().Warn("zenoh: local client connect failed, falling back to discovery",
			zap.String("endpoint", endpoint), zap.Error(err))

		return openDiscovery()
	}

	session, err := openDiscovery()
	if err == nil {
		return session, nil
	}

	logger.Get().Warn("zenoh: discovery failed, falling back to local client connect",
		zap.String("endpoint", endpoint), zap.Error(err))

	return openClient(endpoint)
}

// openClient opens a session in client mode connecting to a single router endpoint.
func openClient(endpoint string) (*zenohSession, error) {
	config := zenoh.NewConfigDefault()

	if err := config.InsertJson5(zenoh.ConfigModeKey, `"client"`); err != nil {
		return nil, fmt.Errorf("zenoh config mode: %w", err)
	}
	if err := config.InsertJson5(zenoh.ConfigConnectKey, fmt.Sprintf(`["%s"]`, endpoint)); err != nil {
		return nil, fmt.Errorf("zenoh config endpoint: %w", err)
	}

	session, err := zenoh.Open(config, nil)
	if err != nil {
		return nil, fmt.Errorf("zenoh open (client): %w", err)
	}

	return &zenohSession{session: session}, nil
}

// openDiscovery opens a session using zenoh's default configuration with network discovery enabled.
func openDiscovery() (*zenohSession, error) {
	session, err := zenoh.Open(zenoh.NewConfigDefault(), nil)
	if err != nil {
		return nil, fmt.Errorf("zenoh open (discovery): %w", err)
	}

	logger.Get().Info("zenoh: session opened with network discovery")
	return &zenohSession{session: session}, nil
}

// DeclarePublisher creates a Publisher for the given key expression.
func (s *zenohSession) DeclarePublisher(key string) (Publisher, error) {
	keyExpr, err := zenoh.NewKeyExpr(key)
	if err != nil {
		return nil, fmt.Errorf("zenoh keyexpr: %w", err)
	}

	pub, err := s.session.DeclarePublisher(keyExpr, nil)
	if err != nil {
		return nil, fmt.Errorf("zenoh declare publisher: %w", err)
	}

	return &zenohPublisher{publisher: pub}, nil
}

// Put publishes data on the publisher's key expression.
func (p *zenohPublisher) Put(data []byte) error {
	if err := p.publisher.Put(zenoh.NewZBytes(data), nil); err != nil {
		return fmt.Errorf("zenoh publisher put: %w", err)
	}

	return nil
}

// Drop undeclares the publisher.
func (p *zenohPublisher) Drop() {
	p.publisher.Drop()
}

// Put publishes data to the given key (one-shot).
func (s *zenohSession) Put(key string, data []byte) error {
	keyExpr, err := zenoh.NewKeyExpr(key)
	if err != nil {
		return fmt.Errorf("zenoh keyexpr: %w", err)
	}

	if err := s.session.Put(keyExpr, zenoh.NewZBytes(data), nil); err != nil {
		return fmt.Errorf("zenoh put: %w", err)
	}

	return nil
}

// DeclareSubscriber creates a Subscriber for the given key expression.
func (s *zenohSession) DeclareSubscriber(key string, handler func([]byte)) (Subscriber, error) {
	keyExpr, err := zenoh.NewKeyExpr(key)
	if err != nil {
		return nil, fmt.Errorf("zenoh keyexpr: %w", err)
	}

	sub, err := s.session.DeclareSubscriber(keyExpr, zenoh.Closure[zenoh.Sample]{
		Call: func(sample zenoh.Sample) { handler(sample.Payload().Bytes()) },
	}, nil)
	if err != nil {
		return nil, fmt.Errorf("zenoh declare subscriber: %w", err)
	}

	return &zenohSubscriber{subscriber: sub}, nil
}

// Drop undeclares the subscriber.
func (sub *zenohSubscriber) Drop() {
	sub.subscriber.Drop()
}

// Close properly closes the zenoh session and notifies the router.
func (s *zenohSession) Close() {
	if err := s.session.Close(nil); err != nil {
		logger.Get().Warn("zenoh: failed to close session", zap.Error(err))
	}
}
