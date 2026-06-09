package zenoh

import (
	"sync"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/cloudsession"
	"github.com/openmind/om1/internal/logger"
)

type hybridSession struct {
	opts Options
	log  *zap.Logger

	mu    sync.Mutex
	cloud Session
	zenoh Session
}

// newHybridSession creates a new hybridSession with the given options.
func newHybridSession(opts Options) *hybridSession {
	if opts.CloudURL == "" {
		opts.CloudURL = cloudsession.DefaultURL
	}
	return &hybridSession{opts: opts, log: logger.Get()}
}

// routeFor determines which backend session to use based on the topic.
func (h *hybridSession) routeFor(topic string) (Session, error) {
	if cloudsession.IsCloudTopic(topic) {
		return h.cloudBackend(), nil
	}
	return h.zenohBackend()
}

// cloudBackend lazily initializes and returns the cloud session.
func (h *hybridSession) cloudBackend() Session {
	h.mu.Lock()
	defer h.mu.Unlock()

	if h.cloud == nil {
		h.cloud = &cloudSession{s: cloudsession.NewSession(h.opts.CloudURL, h.opts.APIKey, h.log)}
		h.log.Info("zenoh: initialized cloud simulation session", zap.String("url", h.opts.CloudURL))
	}

	return h.cloud
}

// zenohBackend lazily initializes and returns the local Zenoh session.
func (h *hybridSession) zenohBackend() (Session, error) {
	h.mu.Lock()
	defer h.mu.Unlock()

	if h.zenoh == nil {
		sess, err := openZenoh(Options{Endpoint: h.opts.Endpoint, LocalNetwork: true})
		if err != nil {
			return nil, err
		}

		h.zenoh = sess
		h.log.Info("zenoh: initialized local session for non-cloud topics")
	}

	return h.zenoh, nil
}

// DeclarePublisher routes the publisher declaration to the appropriate backend session.
func (h *hybridSession) DeclarePublisher(key string) (Publisher, error) {
	sess, err := h.routeFor(key)
	if err != nil {
		return nil, err
	}

	return sess.DeclarePublisher(key)
}

// DeclareSubscriber routes the subscriber declaration to the appropriate backend session.
func (h *hybridSession) DeclareSubscriber(key string, handler func([]byte)) (Subscriber, error) {
	sess, err := h.routeFor(key)
	if err != nil {
		return nil, err
	}

	return sess.DeclareSubscriber(key, handler)
}

// Put routes the publish action to the appropriate backend session.
func (h *hybridSession) Put(key string, data []byte) error {
	sess, err := h.routeFor(key)
	if err != nil {
		return err
	}

	return sess.Put(key, data)
}

// Close properly closes both the cloud and local sessions.
func (h *hybridSession) Close() {
	h.mu.Lock()
	cloud, zenoh := h.cloud, h.zenoh
	h.mu.Unlock()

	if cloud != nil {
		cloud.Close()
		h.log.Info("zenoh: closed cloud simulation session")
	}

	if zenoh != nil {
		zenoh.Close()
		h.log.Info("zenoh: closed local session")
	}
}
