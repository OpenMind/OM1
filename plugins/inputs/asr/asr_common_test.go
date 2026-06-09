package asr

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestCJKRegex(t *testing.T) {
	require.True(t, cjkRegex.MatchString("你好"))
	require.True(t, cjkRegex.MatchString("こんにちは"))
	require.True(t, cjkRegex.MatchString("안녕"))
	require.False(t, cjkRegex.MatchString("hello"))
}

func TestSerializeASRText(t *testing.T) {
	buf := serializeASRText("hi")
	require.Equal(t, []byte{0x00, 0x01, 0x00, 0x00}, buf[:4])
	require.Contains(t, string(buf), "hi")
}
