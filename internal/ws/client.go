package ws

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/gorilla/websocket"
	"go.uber.org/zap"
)

// Config holds connection settings for a Client.
type Config struct {
	URL              string
	HandshakeTimeout time.Duration // default 10s
	WriteTimeout     time.Duration // default 5s
	SendBufferSize   int           // default 256
	Reconnect        bool
}

// Client is a reconnecting WebSocket client for binary streaming.
type Client struct {
	cfg       Config
	log       *zap.Logger
	onMessage func(messageType int, data []byte)

	connMu sync.Mutex
	conn   *websocket.Conn

	ctx    context.Context
	cancel context.CancelFunc

	sendCh chan []byte
	wg     sync.WaitGroup
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
	ctx, cancel := context.WithCancel(context.Background())
	return &Client{
		cfg:       cfg,
		log:       log,
		onMessage: onMessage,
		ctx:       ctx,
		cancel:    cancel,
		sendCh:    make(chan []byte, cfg.SendBufferSize),
	}
}

// Connect dials the server and starts the send/read goroutines.
func (c *Client) Connect() error {
	if !c.cfg.Reconnect {
		return c.dial()
	}

	for {
		err := c.dial()
		if err == nil {
			c.wg.Add(2)
			go c.sendLoop()
			go c.readLoop()
			return nil
		}

		c.log.Warn("ws: initial connect failed, retrying", zap.Error(err))

		select {
		case <-c.ctx.Done():
			return fmt.Errorf("ws: connect cancelled: %w", c.ctx.Err())
		case <-time.After(5 * time.Second):
		}
	}
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
	c.cancel()

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

func (c *Client) dial() error {
	dialer := websocket.Dialer{HandshakeTimeout: c.cfg.HandshakeTimeout}
	c.log.Info("ws: connecting", zap.String("url", c.cfg.URL))

	conn, resp, err := dialer.DialContext(c.ctx, c.cfg.URL, nil)
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

// sendLoop reads from sendCh and writes to the WebSocket.
// On write error, it triggers reconnect if enabled.
func (c *Client) sendLoop() {
	defer c.wg.Done()

	for {
		select {
		case <-c.ctx.Done():
			c.log.Info("ws: send loop stopped")
			return
		case data, ok := <-c.sendCh:
			if !ok {
				return
			}
			if err := c.writeMessage(data); err != nil {
				c.log.Warn("ws: send failed", zap.Error(err))
				if c.cfg.Reconnect {
					c.reconnect()
				}
			}
		}
	}
}

// readLoop reads incoming messages.
// On read error, it triggers reconnect if enabled.
func (c *Client) readLoop() {
	defer c.wg.Done()

	for {
		select {
		case <-c.ctx.Done():
			c.log.Info("ws: read loop stopped")
			return
		default:
		}

		c.connMu.Lock()
		conn := c.conn
		c.connMu.Unlock()
		if conn == nil {
			select {
			case <-c.ctx.Done():
				return
			case <-time.After(50 * time.Millisecond):
			}
			continue
		}

		msgType, msg, err := conn.ReadMessage()
		if err != nil {
			if c.ctx.Err() != nil {
				return
			}
			if websocket.IsUnexpectedCloseError(err,
				websocket.CloseGoingAway, websocket.CloseAbnormalClosure, websocket.CloseNormalClosure) {
				c.log.Warn("ws: read error", zap.Error(err))
			}
			if !c.cfg.Reconnect {
				return
			}
			c.connMu.Lock()
			if c.conn == conn {
				c.conn = nil
			}
			c.connMu.Unlock()
			continue
		}

		if msgType == websocket.TextMessage {
			c.log.Debug("ws: message received",
				zap.String("type", messageTypeName(msgType)),
				zap.String("data", string(msg)),
			)
		} else {
			c.log.Debug("ws: message received",
				zap.String("type", messageTypeName(msgType)),
				zap.ByteString("data", msg),
			)
		}

		if c.onMessage != nil {
			c.onMessage(msgType, msg)
		}
	}
}

func (c *Client) writeMessage(data []byte) error {
	c.connMu.Lock()
	defer c.connMu.Unlock()

	if c.conn == nil {
		return fmt.Errorf("ws: not connected")
	}
	if err := c.conn.SetWriteDeadline(time.Now().Add(c.cfg.WriteTimeout)); err != nil {
		return fmt.Errorf("ws: set deadline: %w", err)
	}
	return c.conn.WriteMessage(websocket.BinaryMessage, data)
}

// reconnect nulls the connection and attempts to re-dial until successful or context cancellation.
func (c *Client) reconnect() {
	c.connMu.Lock()
	if c.conn != nil {
		_ = c.conn.Close()
		c.conn = nil
	}
	c.connMu.Unlock()

	for {
		if c.ctx.Err() != nil {
			return
		}
		if err := c.dial(); err != nil {
			c.log.Error("ws: reconnect failed, retrying", zap.Error(err))
			select {
			case <-c.ctx.Done():
				return
			case <-time.After(5 * time.Second):
			}
			continue
		}
		return
	}
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
