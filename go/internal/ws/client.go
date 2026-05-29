// Package ws provides a reconnecting binary WebSocket client for audio streaming.
// It mirrors the Sender/readLoop/sendLoop pattern from the test_asr reference implementation.
package ws

import (
	"fmt"
	"sync"
	"time"

	"github.com/gorilla/websocket"
	"go.uber.org/zap"
)

// Config holds the connection settings for a Client.
type Config struct {
	URL              string
	HandshakeTimeout time.Duration // default 10s
	WriteTimeout     time.Duration // default 5s
	SendBufferSize   int           // channel depth; default 256
	Reconnect        bool          // auto-reconnect on send failure
}

// Client is a reconnecting WebSocket client for binary streaming.
type Client struct {
	cfg       Config
	log       *zap.Logger
	onMessage func(messageType int, data []byte)

	connMu sync.Mutex
	conn   *websocket.Conn

	sendCh chan []byte
	stopCh chan struct{}
	wg     sync.WaitGroup

	reconnect bool
}

// New creates a Client. onMessage is called for every message received from the server.
func New(cfg Config, log *zap.Logger, onMessage func(messageType int, data []byte)) *Client {
	if cfg.HandshakeTimeout == 0 {
		cfg.HandshakeTimeout = 10 * time.Second
	}
	if cfg.WriteTimeout == 0 {
		cfg.WriteTimeout = 5 * time.Second
	}
	if cfg.SendBufferSize == 0 {
		cfg.SendBufferSize = 256
	}
	return &Client{
		cfg:       cfg,
		log:       log,
		onMessage: onMessage,
		sendCh:    make(chan []byte, cfg.SendBufferSize),
		stopCh:    make(chan struct{}),
		reconnect: cfg.Reconnect,
	}
}

// Connect dials the server and starts the send/read goroutines.
func (c *Client) Connect() error {
	if err := c.dial(); err != nil {
		return err
	}
	c.wg.Add(2)
	go c.sendLoop()
	go c.readLoop()
	return nil
}

// Send enqueues data for delivery. Returns an error if the queue is full.
func (c *Client) Send(data []byte) error {
	select {
	case c.sendCh <- data:
		return nil
	default:
		return fmt.Errorf("ws: send buffer full")
	}
}

// Close performs a graceful shutdown and waits for all goroutines to exit.
func (c *Client) Close() {
	close(c.stopCh)

	c.connMu.Lock()
	if c.conn != nil {
		_ = c.conn.WriteMessage(websocket.CloseMessage,
			websocket.FormatCloseMessage(websocket.CloseNormalClosure, ""))
		_ = c.conn.Close()
		c.conn = nil
	}
	c.connMu.Unlock()

	c.wg.Wait()
}

// dial opens a new WebSocket connection.
func (c *Client) dial() error {
	dialer := websocket.Dialer{HandshakeTimeout: c.cfg.HandshakeTimeout}
	c.log.Info("ws: connecting", zap.String("url", c.cfg.URL))

	conn, resp, err := dialer.Dial(c.cfg.URL, nil)
	if err != nil {
		if resp != nil {
			return fmt.Errorf("ws: connect failed (status %d): %w", resp.StatusCode, err)
		}
		return fmt.Errorf("ws: connect failed: %w", err)
	}
	c.connMu.Lock()
	c.conn = conn
	c.connMu.Unlock()
	c.log.Info("ws: connected", zap.String("url", c.cfg.URL))
	return nil
}

// redial closes the current connection and opens a fresh one, then restarts readLoop.
func (c *Client) redial() error {
	c.connMu.Lock()
	if c.conn != nil {
		_ = c.conn.Close()
		c.conn = nil
	}
	c.connMu.Unlock()

	if err := c.dial(); err != nil {
		return err
	}
	c.wg.Add(1)
	go c.readLoop()
	return nil
}

// sendLoop reads from sendCh and writes binary messages to the WebSocket.
// Mirrors sender.go sendLoop / sendAudioData.
func (c *Client) sendLoop() {
	defer c.wg.Done()
	for {
		select {
		case <-c.stopCh:
			c.log.Info("ws: send loop stopped")
			return
		case data, ok := <-c.sendCh:
			if !ok {
				return
			}
			if err := c.writeMessage(data); err != nil {
				c.log.Warn("ws: send failed", zap.Error(err))

				if c.reconnect {
					c.log.Info("ws: reconnecting...")
					if rerr := c.redial(); rerr != nil {
						c.log.Error("ws: reconnect failed", zap.Error(rerr))
					}
				}
			}
		}
	}
}

// readLoop reads messages from the WebSocket and calls onMessage.
// Mirrors sender.go readLoop.
func (c *Client) readLoop() {
	defer c.wg.Done()
	for {
		c.connMu.Lock()
		conn := c.conn
		c.connMu.Unlock()
		if conn == nil {
			return
		}

		msgType, msg, err := conn.ReadMessage()
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				c.log.Warn("ws: read error", zap.Error(err))
			}
			return
		}
		c.log.Debug("ws: message received",
			zap.String("type", messageTypeName(msgType)),
			zap.Int("bytes", len(msg)),
		)
		if c.onMessage != nil {
			c.onMessage(msgType, msg)
		}
	}
}

func (c *Client) writeMessage(data []byte) error {
	c.connMu.Lock()
	conn := c.conn
	c.connMu.Unlock()
	if conn == nil {
		return fmt.Errorf("ws: not connected")
	}
	if err := conn.SetWriteDeadline(time.Now().Add(c.cfg.WriteTimeout)); err != nil {
		return fmt.Errorf("ws: set deadline: %w", err)
	}
	return conn.WriteMessage(websocket.BinaryMessage, data)
}

func messageTypeName(t int) string {
	switch t {
	case websocket.TextMessage:
		return "text"
	case websocket.BinaryMessage:
		return "binary"
	default:
		return fmt.Sprintf("type(%d)", t)
	}
}
