package greeting_conversation

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestNormalizeTTSTextTimes(t *testing.T) {
	require.Equal(t, "Meet at 11 a.m.", normalizeTTSText("Meet at 11:00 a.m."), "on-the-hour times drop :00")
	require.Equal(t, "Meet at 3 30 p.m.", normalizeTTSText("Meet at 3:30 p.m."), "minutes are spoken separately")
}

func TestNormalizeTTSTextAbbreviations(t *testing.T) {
	cases := map[string]string{
		"See you in Jan": "See you in January",
		"123 Main St":    "123 Main Street",
		"5 Park Ave":     "5 Park Avenue",
		"Sunset Blvd":    "Sunset Boulevard",
	}
	for in, want := range cases {
		t.Run(in, func(t *testing.T) {
			require.Equal(t, want, normalizeTTSText(in))
		})
	}
}

func TestNormalizeTTSTextDirectional(t *testing.T) {
	require.Equal(t, "Go to North Main", normalizeTTSText("Go to N Main"), "directional N before a capitalized word")
}

func TestNormalizeTTSTextNoChange(t *testing.T) {
	require.Equal(t, "Hello there", normalizeTTSText("Hello there"))
}
