package cloudsession

import (
	"encoding/binary"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
)

type brokerServer struct {
	url string

	mu       sync.Mutex
	conn     *websocket.Conn
	controls []map[string]any
	connWG   sync.WaitGroup
}

func newBrokerServer(t *testing.T) *brokerServer {
	t.Helper()
	b := &brokerServer{}
	b.connWG.Add(1)
	upgrader := websocket.Upgrader{CheckOrigin: func(*http.Request) bool { return true }}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}

		b.mu.Lock()
		first := b.conn == nil
		b.conn = conn
		b.mu.Unlock()
		if first {
			b.connWG.Done()
		}

		for {
			mt, msg, err := conn.ReadMessage()
			if err != nil {
				return
			}

			if mt == websocket.TextMessage {
				var parsed map[string]any
				if json.Unmarshal(msg, &parsed) == nil {
					b.mu.Lock()
					b.controls = append(b.controls, parsed)
					b.mu.Unlock()
				}

			} else {
				b.mu.Lock()
				b.controls = append(b.controls, map[string]any{"_binary": msg})
				b.mu.Unlock()
			}
		}
	}))

	t.Cleanup(srv.Close)
	b.url = "ws" + strings.TrimPrefix(srv.URL, "http")
	return b
}

func (b *brokerServer) waitConn(t *testing.T) {
	t.Helper()
	done := make(chan struct{})
	go func() { b.connWG.Wait(); close(done) }()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("client never connected")
	}
}

// pushSample sends a binary sample frame for the given subscription.
func (b *brokerServer) pushSample(subID string, cdr []byte) error {
	frame := []byte{frameSample}
	frame = binary.BigEndian.AppendUint16(frame, uint16(len(subID)))
	frame = append(frame, []byte(subID)...)
	frame = append(frame, cdr...)

	b.mu.Lock()
	conn := b.conn
	b.mu.Unlock()

	return conn.WriteMessage(websocket.BinaryMessage, frame)
}

func (b *brokerServer) controlsCopy() []map[string]any {
	b.mu.Lock()
	defer b.mu.Unlock()

	out := make([]map[string]any, len(b.controls))
	copy(out, b.controls)

	return out
}

func TestWithToken(t *testing.T) {
	require.Equal(t, "ws://x/zenoh?api_key=abc", withToken("ws://x/zenoh", "abc"))
	require.Equal(t, "ws://x/zenoh?a=1&api_key=abc", withToken("ws://x/zenoh?a=1", "abc"))
	require.Equal(t, "ws://x/zenoh", withToken("ws://x/zenoh", ""))
}

func TestDeclareSubscriberReceivesBinarySample(t *testing.T) {
	b := newBrokerServer(t)
	c := NewClient(b.url, "", zap.NewNop())
	t.Cleanup(c.Close)
	b.waitConn(t)

	got := make(chan []byte, 1)
	subID := c.DeclareSubscriber("odom", true, nil, func(data []byte) {
		got <- append([]byte(nil), data...)
	})

	require.Eventually(t, func() bool {
		for _, m := range b.controlsCopy() {
			if m["type"] == "subscribe" && m["id"] == subID && m["topic"] == "odom" {
				return true
			}
		}
		return false
	}, 2*time.Second, 10*time.Millisecond)

	require.NoError(t, b.pushSample(subID, []byte{1, 2, 3, 4}))
	select {
	case data := <-got:
		require.Equal(t, []byte{1, 2, 3, 4}, data)
	case <-time.After(2 * time.Second):
		t.Fatal("did not receive sample")
	}
}

func TestPublishBinaryFrame(t *testing.T) {
	b := newBrokerServer(t)
	c := NewClient(b.url, "", zap.NewNop())
	t.Cleanup(c.Close)
	b.waitConn(t)

	require.NoError(t, c.PublishBinary("cmd_vel", []byte{9, 8, 7}))

	require.Eventually(t, func() bool {
		for _, m := range b.controlsCopy() {
			raw, ok := m["_binary"].([]byte)
			if !ok {
				continue
			}
			topicLen := int(binary.BigEndian.Uint16(raw[1:3]))
			topic := string(raw[3 : 3+topicLen])
			cdr := raw[3+topicLen:]
			if raw[0] == framePublish && topic == "cmd_vel" && string(cdr) == string([]byte{9, 8, 7}) {
				return true
			}
		}
		return false
	}, 2*time.Second, 10*time.Millisecond)
}

func TestUndeclareSubscriber(t *testing.T) {
	b := newBrokerServer(t)
	c := NewClient(b.url, "", zap.NewNop())
	t.Cleanup(c.Close)
	b.waitConn(t)
	require.True(t, c.waitConnected(2*time.Second), "client marks itself connected")

	subID := c.DeclareSubscriber("scan", true, nil, func([]byte) {})
	c.UndeclareSubscriber(subID)

	require.Eventually(t, func() bool {
		for _, m := range b.controlsCopy() {
			if m["type"] == "unsubscribe" && m["id"] == subID {
				return true
			}
		}
		return false
	}, 2*time.Second, 10*time.Millisecond)

	c.subMu.Lock()
	_, exists := c.subs[subID]
	c.subMu.Unlock()
	require.False(t, exists, "subscription is removed locally")
}

func TestPublishBinaryTimesOutWhenNeverConnected(t *testing.T) {
	c := NewClient("ws://127.0.0.1:1/zenoh", "", zap.NewNop())
	t.Cleanup(c.Close)

	done := make(chan error, 1)
	go func() {
		time.AfterFunc(200*time.Millisecond, c.Close)
		done <- c.PublishBinary("odom", []byte{1})
	}()

	select {
	case err := <-done:
		require.Error(t, err)
	case <-time.After(3 * time.Second):
		t.Fatal("PublishBinary blocked when never connected")
	}
}

func TestDispatchBinaryIgnoresMalformedFrames(t *testing.T) {
	c := &Client{log: zap.NewNop(), subs: map[string]*subscription{}}
	require.NotPanics(t, func() {
		c.dispatchBinary(nil)
		c.dispatchBinary([]byte{0x99, 0, 0, 0})
		c.dispatchBinary([]byte{frameSample, 0xFF, 0xFF, 1})
	})
}
