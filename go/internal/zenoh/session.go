package zenoh

import (
	"fmt"

	"github.com/eclipse-zenoh/zenoh-go/zenoh"
)

type Session struct {
	session zenoh.Session
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

// Put publishes data to the given key.
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

// Close closes the zenoh session.
func (s *Session) Close() {
	s.session.Drop()
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
