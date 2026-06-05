package util

import "testing"

func TestFirstNonEmpty(t *testing.T) {
	cases := []struct {
		name string
		in   []string
		want string
	}{
		{"first set", []string{"a", "b"}, "a"},
		{"skips empty", []string{"", "b"}, "b"},
		{"skips whitespace", []string{"   ", "\t", "b"}, "b"},
		{"all empty", []string{"", "  "}, ""},
		{"none", nil, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := FirstNonEmpty(tc.in...); got != tc.want {
				t.Fatalf("FirstNonEmpty(%q) = %q, want %q", tc.in, got, tc.want)
			}
		})
	}
}
