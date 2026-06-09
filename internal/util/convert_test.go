package util

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestToFloat(t *testing.T) {
	cases := []struct {
		name   string
		input  any
		want   float64
		wantOK bool
	}{
		{"float64", float64(1.5), 1.5, true},
		{"float32", float32(2.5), 2.5, true},
		{"int", 3, 3, true},
		{"int64", int64(4), 4, true},
		{"json.Number", json.Number("5.25"), 5.25, true},
		{"invalid json.Number", json.Number("notanumber"), 0, false},
		{"string", "6", 0, false},
		{"nil", nil, 0, false},
		{"bool", true, 0, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := ToFloat(tc.input)
			require.Equal(t, tc.wantOK, ok)
			require.Equal(t, tc.want, got)
		})
	}
}

func TestFloatFrom(t *testing.T) {
	require.Equal(t, 7.5, FloatFrom(7.5, 0))
	require.Equal(t, 9.0, FloatFrom("not a number", 9.0), "falls back to default for non-numeric")
	require.Equal(t, 0.0, FloatFrom(nil, 0.0))
	require.Equal(t, 1.0, FloatFrom(json.Number("bad"), 1.0), "falls back when json.Number is unparsable")
}
