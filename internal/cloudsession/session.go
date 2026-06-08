package cloudsession

import (
	"sync"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/logger"
)

// DefaultURL is the default cloud broker WebSocket endpoint.
const DefaultURL = "wss://api.openmind.com/api/core/simulation/zenoh"

// Subscriber represents an active subscription to a broker topic.
type Subscriber struct {
	session *Session
	topic   string
	subID   string
}

// Drop unsubscribes from the broker topic.
func (s *Subscriber) Drop() {
	s.session.undeclare(s.subID)
}

// Publisher represents a topic publisher to the broker.
type Publisher struct {
	session *Session
	topic   string
}

// Put publishes raw CDR bytes to the broker topic.
func (p *Publisher) Put(payload []byte) error {
	return p.session.publishBinary(p.topic, payload)
}

// Drop is a no-op for Publisher since the cloud broker does not require explicit undeclaration.
func (p *Publisher) Drop() {
	p.session.log.Debug("cloudsession: publisher drop (no-op)", zap.String("topic", p.topic))
}

// Session manages a connection to the cloud broker and provides methods to declare publishers and subscribers.
type Session struct {
	client *Client
	log    *zap.Logger

	mu        sync.Mutex
	subTopics map[string]string
}

// NewSession creates a new Session with the given cloud broker URL and API key.
func NewSession(rawURL, token string, log *zap.Logger) *Session {
	if log == nil {
		log = logger.Get()
	}

	return &Session{
		client:    NewClient(rawURL, token, log),
		log:       log,
		subTopics: make(map[string]string),
	}
}

// DeclareSubscriber returns a subscriber for the given topic and handler function.
func (s *Session) DeclareSubscriber(topic string, handler func([]byte)) *Subscriber {
	onBinary := func(cdr []byte) {
		if handler != nil {
			handler(cdr)
		}
	}
	subID := s.client.DeclareSubscriber(topic, true, nil, onBinary)

	s.mu.Lock()
	s.subTopics[subID] = topic
	s.mu.Unlock()

	return &Subscriber{session: s, topic: topic, subID: subID}
}

// DeclarePublisher returns a publisher whose Put forwards CDR bytes to topic.
func (s *Session) DeclarePublisher(topic string) *Publisher {
	return &Publisher{session: s, topic: topic}
}

// Put publishes raw CDR bytes to the given topic.
func (s *Session) Put(topic string, payload []byte) error {
	return s.publishBinary(topic, payload)
}

// Close closes the underlying broker connection.
func (s *Session) Close() {
	s.client.Close()
}

// undeclare unsubscribes the given subscription ID from the broker and removes it from the session's tracking map.
func (s *Session) undeclare(subID string) {
	s.mu.Lock()
	delete(s.subTopics, subID)
	s.mu.Unlock()
	s.client.UndeclareSubscriber(subID)
}

// publishBinary sends the given CDR bytes to the broker on the specified topic.
func (s *Session) publishBinary(topic string, cdr []byte) error {
	return s.client.PublishBinary(topic, cdr)
}
