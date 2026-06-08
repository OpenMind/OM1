package zenoh

import (
	"github.com/openmind/om1/internal/cloudsession"
)

type cloudSession struct {
	s *cloudsession.Session
}

func (c *cloudSession) DeclarePublisher(key string) (Publisher, error) {
	return c.s.DeclarePublisher(key), nil
}

func (c *cloudSession) DeclareSubscriber(key string, handler func([]byte)) (Subscriber, error) {
	return c.s.DeclareSubscriber(key, handler), nil
}

func (c *cloudSession) Put(key string, data []byte) error { return c.s.Put(key, data) }

func (c *cloudSession) Close() { c.s.Close() }
