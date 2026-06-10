package luma_checkin

import (
	"context"
	"time"

	"go.uber.org/zap"
)

const (
	GetGuestPath            = getGuestPath
	UpdateGuestStatusPath   = updateGuestStatusPath
	DefaultGreetingTemplate = defaultGreetingTemplate
	DefaultStatus           = defaultStatus
)

var (
	ErrLumaNotFound     = errLumaNotFound
	ErrLumaUnauthorized = errLumaUnauthorized
)

type LumaClient = lumaClient

func NewLumaClient(baseURL, apiKey, eventAPIID string, timeout time.Duration) *LumaClient {
	return newLumaClient(baseURL, apiKey, eventAPIID, timeout)
}

func FirstNameFor(g *Guest) string                { return firstNameFor(g) }
func FormatGreeting(tmpl string, g *Guest) string { return formatGreeting(tmpl, g) }

type GuestLookup = guestLookup
type TTSPlayer = ttsPlayer

type TestConnector struct{ *connector }

func NewTestConnector(client GuestLookup, cfg Config, tts TTSPlayer) *TestConnector {
	return &TestConnector{&connector{
		log:    zap.NewNop(),
		cfg:    cfg,
		client: client,
		tts:    tts,
	}}
}

func (c *TestConnector) Connect(ctx context.Context, in map[string]any) (any, error) {
	return c.connector.Connect(ctx, in)
}
