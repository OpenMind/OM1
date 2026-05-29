package zenoh

import (
	"fmt"

	"github.com/eclipse-zenoh/zenoh-go/zenoh"
)

type Session struct {
	session zenoh.Session
}

// Publisher wraps a zenoh Publisher for repeated puts on a fixed key.
type Publisher struct {
	publisher zenoh.Publisher
}

// Open creates a new zenoh session. Optionally accepts a custom endpoint.
func Open(endpoint ...string) (*Session, error) {
	config := zenoh.NewConfigDefault()

	ep := "tcp/127.0.0.1:7447"
	if len(endpoint) > 0 && endpoint[0] != "" {
		ep = endpoint[0]
	}

	if err := config.InsertJson5(zenoh.ConfigModeKey, `"client"`); err != nil {
		return nil, fmt.Errorf("zenoh config mode: %w", err)
	}
	if err := config.InsertJson5(zenoh.ConfigConnectKey, fmt.Sprintf(`["%s"]`, ep)); err != nil {
		return nil, fmt.Errorf("zenoh config endpoint: %w", err)
	}

	session, err := zenoh.Open(config, nil)
	if err != nil {
		return nil, fmt.Errorf("zenoh open: %w", err)
	}

	return &Session{session: session}, nil
}

// DeclarePublisher creates a Publisher for the given key expression.
func (s *Session) DeclarePublisher(key string) (*Publisher, error) {
	keyExpr, err := zenoh.NewKeyExpr(key)
	if err != nil {
		return nil, fmt.Errorf("zenoh keyexpr: %w", err)
	}
	pub, err := s.session.DeclarePublisher(keyExpr, nil)
	if err != nil {
		return nil, fmt.Errorf("zenoh declare publisher: %w", err)
	}
	return &Publisher{publisher: pub}, nil
}

// Put publishes data on the publisher's key expression.
func (p *Publisher) Put(data []byte) error {
	if err := p.publisher.Put(zenoh.NewZBytes(data), nil); err != nil {
		return fmt.Errorf("zenoh publisher put: %w", err)
	}
	return nil
}

// Drop undeclares the publisher.
func (p *Publisher) Drop() {
	p.publisher.Drop()
}

// Put publishes data to the given key (one-shot).
func (s *Session) Put(key string, data []byte) error {
	keyExpr, err := zenoh.NewKeyExpr(key)
	if err != nil {
		return fmt.Errorf("zenoh keyexpr: %w", err)
	}

	if err := s.session.Put(keyExpr, zenoh.NewZBytes(data), nil); err != nil {
		return fmt.Errorf("zenoh put: %w", err)
	}

	return nil
}

// Close properly closes the zenoh session and notifies the router.
func (s *Session) Close() {
	s.session.Close(nil) //nolint:errcheck
}

// Publish opens a session, publishes data, and closes the session.
func Publish(key string, data []byte, endpoint ...string) error {
	session, err := Open(endpoint...)
	if err != nil {
		return err
	}
	defer session.Close()

	return session.Put(key, data)
}
