package inputs

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/inputs"
	"github.com/openmind/om1/internal/logger"
	"github.com/openmind/om1/internal/providers"
)

func init() {
	inputs.Register("ConversationHistoryInput", NewConversationHistory)
}

const (
	conversationHistoryDescriptor    = "Conversation History"
	conversationHistoryVoiceKey      = "Voice"
	conversationHistoryPollInterval  = 500 * time.Millisecond
	conversationHistoryDefaultRounds = 3
)

// ConversationHistoryConfig is the JSON configuration for the ConversationHistory sensor.
type ConversationHistoryConfig struct {
	MaxRounds int `json:"max_rounds"`
}

// ConversationHistorySensor polls the IOProvider for voice inputs and maintains a
// sliding window of conversation history for the LLM prompt.
type ConversationHistorySensor struct {
	cfg      ConversationHistoryConfig
	log      *zap.Logger
	provider *providers.IOProvider

	mu               sync.Mutex
	messages         []inputs.Message
	lastRecordedTick int
	stopped          bool
}

// NewConversationHistory constructs a ConversationHistorySensor with the given configuration.
func NewConversationHistory(configMap map[string]any) (inputs.Sensor, error) {
	var cfg ConversationHistoryConfig
	if b, err := json.Marshal(configMap); err == nil {
		_ = json.Unmarshal(b, &cfg)
	}
	if cfg.MaxRounds <= 0 {
		cfg.MaxRounds = conversationHistoryDefaultRounds
	}

	log := logger.Get().Named("ConversationHistoryInput")
	log.Info("initializing",
		zap.Int("max_rounds", cfg.MaxRounds),
	)

	return &ConversationHistorySensor{
		cfg:              cfg,
		log:              log,
		provider:         providers.IO(),
		lastRecordedTick: -1,
	}, nil
}

// Listen polls the IOProvider for voice inputs at a fixed cadence and yields each
// new voice line on the returned channel until ctx is cancelled.
func (s *ConversationHistorySensor) Listen(ctx context.Context) (<-chan any, error) {
	out := make(chan any)
	go func() {
		defer close(out)
		defer s.Stop()

		ticker := time.NewTicker(conversationHistoryPollInterval)
		defer ticker.Stop()

		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
			}

			raw, err := s.Poll(ctx)
			if err != nil || raw == nil {
				continue
			}

			select {
			case out <- raw:
			case <-ctx.Done():
				return
			}
		}
	}()
	return out, nil
}

// Poll checks the IOProvider for new voice input on the current tick. It returns
// the voice text if new, or nil otherwise.
func (s *ConversationHistorySensor) Poll(_ context.Context) (any, error) {
	s.mu.Lock()
	if s.stopped {
		s.mu.Unlock()
		return nil, nil
	}
	lastRecorded := s.lastRecordedTick
	s.mu.Unlock()

	currentTick := s.provider.TickCounter()
	if currentTick <= lastRecorded {
		return nil, nil
	}

	voiceInput := s.provider.GetInput(conversationHistoryVoiceKey)
	if voiceInput != nil && voiceInput.Input != "" && voiceInput.Tick == currentTick {
		text := strings.TrimSpace(voiceInput.Input)
		if text != "" {
			s.mu.Lock()
			s.lastRecordedTick = currentTick
			s.mu.Unlock()
			return text, nil
		}
	}

	return nil, nil
}

// RawToText converts a raw voice line into a timestamped Message and appends it to
// the bounded sliding-window history.
func (s *ConversationHistorySensor) RawToText(_ context.Context, raw any) (*inputs.Message, error) {
	text, ok := raw.(string)
	if !ok || text == "" {
		return nil, nil
	}

	msg := inputs.NewMessage(text)

	s.mu.Lock()
	s.messages = append(s.messages, *msg)
	if len(s.messages) > s.cfg.MaxRounds {
		s.messages = s.messages[len(s.messages)-s.cfg.MaxRounds:]
	}
	s.mu.Unlock()

	return msg, nil
}

// FormattedLatestBuffer returns the current conversation history as a single formatted string
// for LLM prompting. Each line is prefixed with "User: " and separated by "; ".
func (s *ConversationHistorySensor) FormattedLatestBuffer() string {
	s.mu.Lock()
	defer s.mu.Unlock()

	if len(s.messages) == 0 {
		return ""
	}

	lines := make([]string, len(s.messages))
	for i, msg := range s.messages {
		lines[i] = fmt.Sprintf("\n%s : %s\n", conversationHistoryDescriptor, msg.Message)
	}

	return fmt.Sprintf("%s: %q", conversationHistoryDescriptor, strings.Join(lines, "; "))
}

// Stop clears the message history and resets state. The polling goroutine
// terminates via context cancellation in Listen.
func (s *ConversationHistorySensor) Stop() {
	s.mu.Lock()
	if s.stopped {
		s.mu.Unlock()
		return
	}
	s.stopped = true
	s.messages = nil
	s.mu.Unlock()

	s.log.Info("stopping sensor")
}
