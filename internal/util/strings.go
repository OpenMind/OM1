package util

import "strings"

// FirstNonEmpty returns the first non-empty string from the provided list, or an empty string if all are empty.
func FirstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if strings.TrimSpace(v) != "" {
			return v
		}
	}
	return ""
}

// TrimLower trims whitespace and converts the string to lowercase.
func TrimLower(s string) string {
	return strings.ToLower(strings.TrimSpace(s))
}
