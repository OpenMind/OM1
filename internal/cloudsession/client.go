package cloudsession

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"net/url"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/gorilla/websocket"
	"go.uber.org/zap"

	"github.com/openmind/om1/internal/logger"
)

const maxMessageSize = 32 * 1024 * 1024

const (
	minBackoff = 500 * time.Millisecond
	maxBackoff = 30 * time.Second
)

const (
	framePublish = 0x20 // client -> server: [0x20][2B topic len BE][topic][cdr]
	frameSample  = 0x10 // server -> client: [0x10][2B subId len BE][subId][cdr]
)

type PayloadCallback func(payload map[string]any)

type BinaryCallback func(data []byte)

type subscription struct {
	topic    string
	binary   bool
	onJSON   PayloadCallback
	onBinary BinaryCallback
}

type Client struct {
	url string
	log *zap.Logger

	mu   sync.Mutex
	conn *websocket.Conn

	subMu sync.Mutex
	subs  map[string]*subscription

	connectedMu sync.Mutex
	connected   chan struct{}

	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

// NewClient creates a new Client and starts the connection loop.
func NewClient(rawURL, token string, log *zap.Logger) *Client {
	if log == nil {
		log = logger.Get()
	}
	ctx, cancel := context.WithCancel(context.Background())
	c := &Client{
		url:       withToken(rawURL, token),
		log:       log,
		subs:      make(map[string]*subscription),
		connected: make(chan struct{}),
		ctx:       ctx,
		cancel:    cancel,
	}

	c.wg.Add(1)
	go c.run()
	return c
}

// withToken appends the token as an api_key query parameter if non-empty, preserving existing parameters.
func withToken(rawURL, token string) string {
	if token == "" {
		return rawURL
	}

	sep := "?"
	if u, err := url.Parse(rawURL); err == nil && u.RawQuery != "" {
		sep = "&"
	} else if err != nil {
		for i := 0; i < len(rawURL); i++ {
			if rawURL[i] == '?' {
				sep = "&"
				break
			}
		}
	}

	return fmt.Sprintf("%s%sapi_key=%s", rawURL, sep, url.QueryEscape(token))
}

// DeclareSubscriber subscribes handler to JSON messages on topic.
func (c *Client) DeclareSubscriber(topic string, binary bool, onJSON PayloadCallback, onBinary BinaryCallback) string {
	subID := uuid.New().String()[:12]
	sub := &subscription{topic: topic, binary: binary, onJSON: onJSON, onBinary: onBinary}

	c.subMu.Lock()
	c.subs[subID] = sub
	c.subMu.Unlock()

	c.sendJSON(map[string]any{
		"type":   "subscribe",
		"id":     subID,
		"topic":  topic,
		"binary": binary,
	})
	return subID
}

// UndeclareSubscriber cancels the subscription identified by subID.
func (c *Client) UndeclareSubscriber(subID string) {
	c.subMu.Lock()
	delete(c.subs, subID)
	c.subMu.Unlock()

	c.sendJSON(map[string]any{"type": "unsubscribe", "id": subID})
}

// Publish sends a JSON payload; the server encodes it to CDR.
func (c *Client) Publish(topic string, payload map[string]any) {
	c.sendJSON(map[string]any{"type": "publish", "topic": topic, "payload": payload})
}

// PublishBinary sends pre-encoded CDR bytes using the binary frame format:
// [0x20][2B topic len BE][topic UTF-8][CDR bytes].
func (c *Client) PublishBinary(topic string, cdr []byte) error {
	topicBytes := []byte(topic)
	if len(topicBytes) > 0xFFFF {
		return fmt.Errorf("cloudsession: topic too long")
	}
	frame := make([]byte, 0, 3+len(topicBytes)+len(cdr))
	frame = append(frame, framePublish)
	frame = binary.BigEndian.AppendUint16(frame, uint16(len(topicBytes)))
	frame = append(frame, topicBytes...)
	frame = append(frame, cdr...)
	return c.sendBytes(frame)
}

// Close stops the reconnect loop and closes the underlying WebSocket.
func (c *Client) Close() {
	c.cancel()

	c.mu.Lock()
	if c.conn != nil {
		_ = c.conn.WriteMessage(websocket.CloseMessage,
			websocket.FormatCloseMessage(websocket.CloseNormalClosure, ""))
		_ = c.conn.Close()
	}
	c.mu.Unlock()

	done := make(chan struct{})
	go func() { c.wg.Wait(); close(done) }()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		c.log.Warn("cloudsession: timed out waiting for client to stop")
	}
}

// sendJSON marshals and writes a control message as a text frame.
func (c *Client) sendJSON(msg map[string]any) {
	data, err := json.Marshal(msg)
	if err != nil {
		c.log.Warn("cloudsession: marshal message", zap.Error(err))
		return
	}

	c.mu.Lock()
	defer c.mu.Unlock()
	if c.conn == nil {
		return
	}

	if err := c.conn.WriteMessage(websocket.TextMessage, data); err != nil {
		c.log.Warn("cloudsession: send failed", zap.Error(err))
	}
}

// sendBytes writes a binary frame, waiting for connection if needed.
func (c *Client) sendBytes(data []byte) error {
	if !c.waitConnected(30 * time.Second) {
		return fmt.Errorf("cloudsession: not connected: timeout waiting for connection")
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	if c.conn == nil {
		return fmt.Errorf("cloudsession: not connected")
	}

	return c.conn.WriteMessage(websocket.BinaryMessage, data)
}

// run manages the connection lifecycle: dial, resubscribe, read until error, then reconnect with exponential backoff.
func (c *Client) run() {
	defer c.wg.Done()

	dialer := websocket.Dialer{
		HandshakeTimeout: 10 * time.Second,
		ReadBufferSize:   0,
	}
	backoff := minBackoff

	for c.ctx.Err() == nil {
		conn, _, err := dialer.DialContext(c.ctx, c.url, nil)
		if err != nil {
			if c.ctx.Err() != nil {
				return
			}
			c.log.Warn("cloudsession: cloud WS dropped — reconnecting",
				zap.Error(err), zap.Duration("backoff", backoff))
		} else {
			conn.SetReadLimit(maxMessageSize)
			c.mu.Lock()
			c.conn = conn
			c.mu.Unlock()

			backoff = minBackoff
			c.resubscribe()
			c.setConnected(true)

			c.readMessages(conn)

			c.setConnected(false)
			c.mu.Lock()
			if c.conn == conn {
				c.conn = nil
			}
			c.mu.Unlock()
			_ = conn.Close()
		}

		if c.ctx.Err() != nil {
			return
		}
		select {
		case <-c.ctx.Done():
			return
		case <-time.After(backoff):
		}
		backoff = min(backoff*2, maxBackoff)
	}
}

// resubscribe re-sends all active subscriptions after a (re)connect.
func (c *Client) resubscribe() {
	c.subMu.Lock()
	specs := make([]map[string]any, 0, len(c.subs))
	for subID, sub := range c.subs {
		specs = append(specs, map[string]any{
			"type":   "subscribe",
			"id":     subID,
			"topic":  sub.topic,
			"binary": sub.binary,
		})
	}
	c.subMu.Unlock()

	for _, spec := range specs {
		c.sendJSON(spec)
	}
}

// readMessages reads frames until the connection errors or the client stops.
func (c *Client) readMessages(conn *websocket.Conn) {
	for {
		if c.ctx.Err() != nil {
			return
		}
		msgType, raw, err := conn.ReadMessage()
		if err != nil {
			if c.ctx.Err() == nil {
				c.log.Warn("cloudsession: read error", zap.Error(err))
			}
			return
		}

		if msgType == websocket.BinaryMessage {
			c.dispatchBinary(raw)
			continue
		}

		var msg map[string]any
		if err := json.Unmarshal(raw, &msg); err != nil {
			c.log.Warn("cloudsession: server sent non-JSON text frame")
			continue
		}
		c.dispatchJSON(msg)
	}
}

// dispatchJSON routes a JSON server message to the appropriate callback.
func (c *Client) dispatchJSON(msg map[string]any) {
	msgType, _ := msg["type"].(string)
	switch msgType {
	case "sample":
		subID, _ := msg["subId"].(string)
		if subID == "" {
			return
		}

		c.subMu.Lock()
		sub := c.subs[subID]
		c.subMu.Unlock()
		if sub == nil || sub.onJSON == nil {
			return
		}

		payload, _ := msg["payload"].(map[string]any)
		if payload == nil {
			payload = map[string]any{}
		}

		c.safeJSON(sub.onJSON, payload)
	case "ack":
		c.log.Debug("cloudsession: ack", zap.Any("id", msg["id"]))
	case "error":
		c.log.Warn("cloudsession: server error", zap.Any("msg", msg))
	case "pong":
	default:
		c.log.Warn("cloudsession: unknown server message type", zap.String("type", msgType))
	}
}

// dispatchBinary routes a binary sample frame to the subscriber callback.
func (c *Client) dispatchBinary(frame []byte) {
	if len(frame) < 4 || frame[0] != frameSample {
		marker := -1
		if len(frame) > 0 {
			marker = int(frame[0])
		}
		c.log.Warn("cloudsession: unknown binary frame", zap.Int("type", marker))
		return
	}

	subIDLen := int(binary.BigEndian.Uint16(frame[1:3]))
	if 3+subIDLen > len(frame) {
		c.log.Warn("cloudsession: malformed binary sample frame")
		return
	}
	cdr := frame[3+subIDLen:]

	c.subMu.Lock()
	sub := c.subs[string(frame[3:3+subIDLen])]
	c.subMu.Unlock()
	if sub == nil || sub.onBinary == nil {
		return
	}
	c.safeBinary(sub.onBinary, cdr)
}

// safeJSON calls the JSON callback, recovering and logging any panics.
func (c *Client) safeJSON(cb PayloadCallback, payload map[string]any) {
	defer func() {
		if r := recover(); r != nil {
			c.log.Error("cloudsession: subscriber callback panicked", zap.Any("recover", r))
		}
	}()
	cb(payload)
}

// safeBinary calls the binary callback, recovering and logging any panics.
func (c *Client) safeBinary(cb BinaryCallback, data []byte) {
	defer func() {
		if r := recover(); r != nil {
			c.log.Error("cloudsession: binary subscriber callback panicked", zap.Any("recover", r))
		}
	}()
	cb(data)
}

// setConnected opens or closes the connected signal channel.
func (c *Client) setConnected(connected bool) {
	c.connectedMu.Lock()
	defer c.connectedMu.Unlock()

	if connected {
		select {
		case <-c.connected:
		default:
			close(c.connected)
		}
		return
	}

	select {
	case <-c.connected:
		c.connected = make(chan struct{})
	default:
	}
}

// waitConnected blocks until connected, the timeout elapses, or the client stops.
func (c *Client) waitConnected(timeout time.Duration) bool {
	c.connectedMu.Lock()
	ch := c.connected
	c.connectedMu.Unlock()

	select {
	case <-ch:
		return true
	case <-c.ctx.Done():
		return false
	case <-time.After(timeout):
		return false
	}
}
