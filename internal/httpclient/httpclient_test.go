package httpclient

import (
	"net/http"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestDefaultReturnsSharedSingleton(t *testing.T) {
	c1 := Default()
	c2 := Default()
	require.NotNil(t, c1)
	require.Same(t, c1, c2, "Default returns the same process-wide client")
}

func TestDefaultTransportTuning(t *testing.T) {
	transport, ok := Default().Transport.(*http.Transport)
	require.True(t, ok, "the default client uses an *http.Transport")
	require.Equal(t, 500, transport.MaxIdleConns)
	require.Equal(t, 100, transport.MaxIdleConnsPerHost)
	require.True(t, transport.ForceAttemptHTTP2)
}
