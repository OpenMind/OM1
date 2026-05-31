package util

import "encoding/json"

// FloatFrom extracts a float64 from a JSON-decoded value, returning def when the
// value is absent or not a recognized numeric type.
func FloatFrom(v any, def float64) float64 {
	if f, ok := ToFloat(v); ok {
		return f
	}
	return def
}

// ToFloat coerces a JSON-decoded numeric value to float64. The bool is false
// when v is not a recognized numeric type.
func ToFloat(v any) (float64, bool) {
	switch n := v.(type) {
	case float64:
		return n, true
	case float32:
		return float64(n), true
	case int:
		return float64(n), true
	case int64:
		return float64(n), true
	case json.Number:
		f, err := n.Float64()
		return f, err == nil
	default:
		return 0, false
	}
}
