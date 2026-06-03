package ws

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
)

func echoServer(t *testing.T) string {
	t.Helper()
	upgrader := websocket.Upgrader{CheckOrigin: func(*http.Request) bool { return true }}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer conn.Close()
		for {
			mt, msg, err := conn.ReadMessage()
			if err != nil {
				return
			}
			if err := conn.WriteMessage(mt, msg); err != nil {
				return
			}
		}
	}))
	t.Cleanup(srv.Close)
	return "ws" + strings.TrimPrefix(srv.URL, "http")
}

func TestNewAppliesDefaults(t *testing.T) {
	c := New(Config{URL: "ws://x"}, zap.NewNop(), nil)
	require.Equal(t, 10*time.Second, c.cfg.HandshakeTimeout)
	require.Equal(t, 5*time.Second, c.cfg.WriteTimeout)
	require.Equal(t, 256, c.cfg.SendBufferSize)
	require.Equal(t, 256, cap(c.sendCh))
}

func TestSendBufferFull(t *testing.T) {
	c := New(Config{URL: "ws://x", SendBufferSize: 1}, zap.NewNop(), nil)
	require.NoError(t, c.Send([]byte("a")), "first send fits in the buffer")
	require.Error(t, c.Send([]byte("b")), "second send overflows the buffer")
}

func TestConnectSendReceive(t *testing.T) {
	url := echoServer(t)

	received := make(chan []byte, 1)
	c := New(Config{URL: url, Reconnect: true}, zap.NewNop(), func(_ int, data []byte) {
		received <- append([]byte(nil), data...)
	})
	require.NoError(t, c.Connect())
	t.Cleanup(c.Close)

	require.NoError(t, c.Send([]byte("ping")))
	select {
	case got := <-received:
		require.Equal(t, []byte("ping"), got, "the server echoes the message back to onMessage")
	case <-time.After(2 * time.Second):
		t.Fatal("did not receive echoed message")
	}
}

func TestConnectFailsForBadURL(t *testing.T) {
	c := New(Config{URL: "ws://127.0.0.1:1"}, zap.NewNop(), nil)
	require.Error(t, c.Connect(), "non-reconnecting client returns the dial error")
}

func TestMessageTypeName(t *testing.T) {
	require.Equal(t, "text", messageTypeName(websocket.TextMessage))
	require.Equal(t, "binary", messageTypeName(websocket.BinaryMessage))
	require.Contains(t, messageTypeName(99), "type(99)")
}

func TestCloseIsIdempotentBeforeConnect(t *testing.T) {
	c := New(Config{URL: "ws://x"}, zap.NewNop(), nil)
	require.NotPanics(t, c.Close, "Close is safe even if Connect was never called")
}
