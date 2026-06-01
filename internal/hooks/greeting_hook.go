package hooks

import (
	"context"
	"fmt"
	"strings"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/providers"
)

func init() {
	RegisterHook("greeting_hook", "greeting_start_hook", (*Runner).greetingStartHook)
	RegisterHook("greeting_hook", "greeting_end_hook", (*Runner).greetingEndHook)
}

// greetingStartHook handles the start of a greeting conversation. It announces a
// personalized greeting, using face presence to address a recognized person by
// name when one is in view.
func (r *Runner) greetingStartHook(ctx context.Context, cfg, vars map[string]any) error {
	provider, err := r.greetingTTSProvider(cfg)
	if err != nil {
		r.log.Error("greeting_start_hook: error", zap.Error(err))
		return err
	}

	robotIntro := ""
	if name := formatTemplate(stringVal(cfg, "robot_name"), vars); name != "" {
		robotIntro = fmt.Sprintf("I'm %s. ", name)
	}

	helpMessage := "How can I help you today?"
	if custom := formatTemplate(stringVal(cfg, "custom_message"), vars); custom != "" {
		helpMessage = custom
	}

	face := providers.NewFacePresenceProvider(providers.FacePresenceConfig{})
	snapshot, snapErr := face.FetchSnapshot(ctx)

	switch {
	case snapErr != nil:
		r.log.Warn("greeting_start_hook: face snapshot failed", zap.Error(snapErr))
		provider.AddText(fmt.Sprintf("Hello! %s%s", robotIntro, helpMessage))
	case !strings.EqualFold(snapshot.ClosestName, "unknown"):
		provider.AddText(fmt.Sprintf("Hello %s! %sNice to see you. %s", snapshot.ClosestName, robotIntro, helpMessage))
	default:
		provider.AddText(fmt.Sprintf("Hello! %s%s", robotIntro, helpMessage))
	}

	r.log.Info("greeting start hook executed successfully")
	return nil
}

// greetingEndHook handles the end of a greeting conversation, announcing a
// farewell whose wording depends on how far the conversation progressed.
func (r *Runner) greetingEndHook(_ context.Context, cfg, _ map[string]any) error {
	provider, err := r.greetingTTSProvider(cfg)
	if err != nil {
		r.log.Error("greeting_end_hook: error", zap.Error(err))
		return err
	}

	state := providers.Greeting()
	switch {
	case state.TurnCount() >= state.MaxTurnCount():
		r.log.Info("greeting conversation ended due to maximum turn count")
		provider.AddText("Thank you for chatting with me today. I hope you enjoy the rest of your day.")
	case state.TurnCount() > 0:
		provider.AddText("It was nice talking with you! If you have any more questions, come chat with me again!")
	default:
		provider.AddText("It was great meeting you! If you want to chat later, just come back and say hi!")
	}

	return nil
}

// greetingTTSProvider resolves the TTS provider for a greeting hook.
func (r *Runner) greetingTTSProvider(cfg map[string]any) (*providers.ElevenLabsProvider, error) {
	tts := strings.ToLower(stringVal(cfg, "tts_provider"))

	if tts != "" && tts != "elevenlabs" {
		r.log.Warn("greeting hook: TTS provider not supported, falling back to elevenlabs",
			zap.String("tts_provider", tts))
	}

	return providers.ElevenLabs(elevenLabsConfigFrom(cfg), r.log), nil
}
