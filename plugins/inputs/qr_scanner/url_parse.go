package qr_scanner

import (
	"net/url"
	"strings"
)

// parseLumaCheckinURL extracts the event ID and pk from a Luma check-in URL of
// the form https://luma.com/check-in/<eventID>?pk=<pk>. Accepted hosts are
// luma.com, www.luma.com, and lu.ma.
func parseLumaCheckinURL(s string) (eventID, pk string, ok bool) {
	u, err := url.Parse(strings.TrimSpace(s))
	if err != nil {
		return "", "", false
	}
	if u.Scheme != "http" && u.Scheme != "https" {
		return "", "", false
	}
	host := strings.ToLower(u.Hostname())
	if host != "luma.com" && host != "www.luma.com" && host != "lu.ma" {
		return "", "", false
	}
	parts := strings.Split(strings.Trim(u.Path, "/"), "/")
	if len(parts) != 2 || parts[0] != "check-in" || parts[1] == "" {
		return "", "", false
	}
	pk = u.Query().Get("pk")
	if pk == "" {
		return "", "", false
	}
	return parts[1], pk, true
}
