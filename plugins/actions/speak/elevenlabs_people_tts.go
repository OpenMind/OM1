package speak

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/actions"
	"github.com/openmind/om1/internal/logger"
	"github.com/openmind/om1/internal/providers"
	"github.com/openmind/om1/internal/providers/tts"
)

func init() {
	actions.Register("speak/elevenlabs_people_tts", NewElevenLabsPeopleTTS)
}

const facePresenceIOKey = "FacePresence"

// closestPersonRe extracts the closest person's name from a FacePresence input
var closestPersonRe = regexp.MustCompile(`Closest:\s*(\w+)`)

// ElevenLabsPeopleConnector is an extension of ElevenLabsConnector
// that selects the voice based on the closest person in view
type ElevenLabsPeopleConnector struct {
	*ElevenLabsConnector

	defaultVoiceID string
	voiceIDs       map[string]string
}

// NewElevenLabsPeopleTTS creates a new ElevenLabsPeopleConnector with the provided configuration.
func NewElevenLabsPeopleTTS(configMap map[string]any) (actions.Connector, error) {
	base, err := NewElevenLabsTTS(configMap)
	if err != nil {
		return nil, err
	}
	baseConn, ok := base.(*ElevenLabsConnector)
	if !ok {
		return nil, fmt.Errorf("speak/elevenlabs_people_tts: unexpected base connector type %T", base)
	}

	var cfg struct {
		VoiceID  string            `json:"voice_id"`
		VoiceIDs map[string]string `json:"voice_ids"`
	}
	if b, err := json.Marshal(configMap); err == nil {
		_ = json.Unmarshal(b, &cfg)
	}
	if cfg.VoiceID == "" {
		cfg.VoiceID = tts.DefaultVoiceID
	}

	baseConn.log = logger.Get().Named("speak/elevenlabs_people_tts")

	return &ElevenLabsPeopleConnector{
		ElevenLabsConnector: baseConn,
		defaultVoiceID:      cfg.VoiceID,
		voiceIDs:            cfg.VoiceIDs,
	}, nil
}

// Connect implements the Connector interface. It extracts the text to speak from the input,
// determines the appropriate voice based on the closest person in view, and enqueues the text
// for speech synthesis with ElevenLabs.
func (e *ElevenLabsPeopleConnector) Connect(_ context.Context, input actions.Input) (actions.Output, error) {
	arguments, ok := input.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("speak/elevenlabs_people_tts: unexpected input type %T", input)
	}
	text, _ := arguments["action"].(string)
	if text == "" {
		return nil, nil
	}

	e.silenceMu.Lock()
	if e.silenceRate > 0 && e.silenceCounter < e.silenceRate {
		e.silenceCounter++
		e.silenceMu.Unlock()
		e.log.Info("skipping (silence_rate)", zap.Int("counter", e.silenceCounter))
		return nil, nil
	}
	e.silenceCounter = 0
	e.silenceMu.Unlock()

	voiceID := e.resolveVoiceID()
	e.log.Info("enqueueing text",
		zap.String("text", text),
		zap.String("voice_id", voiceID),
	)
	e.elevenlabs.AddTextWithVoice(text, voiceID)

	return nil, nil
}

// resolveVoiceID determines which voice ID to use based on the closest person detected by the FacePresence sensor.
func (e *ElevenLabsPeopleConnector) resolveVoiceID() string {
	io := providers.IO()

	peopleName := ""
	if in := io.GetInput(facePresenceIOKey); in != nil && in.Input != "" && in.Tick == io.TickCounter() {
		if m := closestPersonRe.FindStringSubmatch(in.Input); m != nil {
			name := strings.TrimSpace(m[1])
			if !strings.EqualFold(name, "unknown") {
				peopleName = name
			}
		}
	}

	if peopleName != "" && e.voiceIDs != nil {
		if voiceID, ok := e.voiceIDs[peopleName]; ok {
			e.log.Info("matched person voice",
				zap.String("people", peopleName),
				zap.String("voice_id", voiceID),
			)
			return voiceID
		}
	}

	return e.defaultVoiceID
}
