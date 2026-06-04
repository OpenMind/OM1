package emotion

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestEmotionEnumValues(t *testing.T) {
	values := Emotion("").EnumValues()
	require.Contains(t, values, "happy")
	require.Contains(t, values, "sad")
	require.Len(t, values, 6, "all six supported emotions are advertised")
}
