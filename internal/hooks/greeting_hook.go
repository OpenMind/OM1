package hooks

import (
	"context"
	"fmt"
	"strings"
	"time"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/llm"
	"github.com/openmind/om1/internal/providers"
)

func init() {
	RegisterHook("greeting_hook", "greeting_start_hook", (*Runner).greetingStartHook)
	RegisterHook("greeting_hook", "greeting_end_hook", (*Runner).greetingEndHook)
}

const defaultGreetingLLMType = "GeminiLLM"

const defaultGreetingPrompt = "You are {robot_name}, a friendly robot greeting whoever is in front of you. " +
	"The current time is {current_time}. " +
	"Generate a single warm, natural spoken greeting of one or two short sentences. " +
	"Here is what you currently see: {scene}. " +
	"If a specific person is recognized ({closest_name}), greet them by name; otherwise greet generically. " +
	"You may reflect the time of day in your greeting when it feels natural. " +
	"Finish by offering help, for example: \"{help_message}\". " +
	"Respond with only the greeting text, with no quotes or commentary."

// greetingStartHook handles the start of a greeting conversation by generating a
// greeting message using an LLM and sending it to the TTS provider.
func (r *Runner) greetingStartHook(ctx context.Context, cfg, vars map[string]any) error {
	provider, err := r.greetingTTSProvider(cfg)
	if err != nil {
		r.log.Error("greeting_start_hook: error", zap.Error(err))
		return err
	}

	robotName := formatTemplate(stringVal(cfg, "robot_name"), vars)

	helpMessage := "How can I help you today?"
	if custom := formatTemplate(stringVal(cfg, "custom_message"), vars); custom != "" {
		helpMessage = custom
	}

	face := providers.NewFacePresenceProvider(providers.FacePresenceConfig{})
	snapshot, snapErr := face.FetchSnapshot(ctx)
	if snapErr != nil {
		r.log.Warn("greeting_start_hook: face snapshot failed", zap.Error(snapErr))
	}

	if greeting, genErr := r.generateGreeting(ctx, cfg, vars, snapshot, robotName, helpMessage); genErr != nil {
		r.log.Warn("greeting_start_hook: llm generation failed, using static greeting", zap.Error(genErr))
		provider.AddText(staticGreeting(snapshot, snapErr, robotName, helpMessage))
	} else {
		r.log.Info("greeting generated successfully", zap.String("greeting", greeting))
		provider.AddText(greeting)
	}

	r.log.Info("greeting start hook executed successfully")
	return nil
}

// generateGreeting builds the prompt from the current scene and asks the
// configured LLM for a spoken greeting, returning the trimmed generated text.
// ToDo:
// - add more context to the prompt, e.g. recent conversation history if available, etc.
func (r *Runner) generateGreeting(ctx context.Context, cfg, vars map[string]any, snapshot providers.PresenceSnapshot, robotName, helpMessage string) (string, error) {
	model, err := r.greetingLLM(cfg)
	if err != nil {
		return "", err
	}

	if robotName == "" {
		robotName = "a friendly robot"
	}

	scene := snapshot.ToText()
	if strings.TrimSpace(scene) == "" {
		scene = "No one is clearly in view."
	}

	closestName := snapshot.ClosestName
	if closestName == "" || strings.EqualFold(closestName, "unknown") {
		closestName = "no one in particular"
	}

	promptTemplate := stringVal(cfg, "prompt")
	if strings.TrimSpace(promptTemplate) == "" {
		promptTemplate = defaultGreetingPrompt
	}

	promptVars := make(map[string]any, len(vars)+5)
	for k, v := range vars {
		promptVars[k] = v
	}
	promptVars["robot_name"] = robotName
	promptVars["scene"] = scene
	promptVars["closest_name"] = closestName
	promptVars["help_message"] = helpMessage
	promptVars["current_time"] = time.Now().Format("Monday, January 2, 2006 at 3:04 PM")

	prompt := formatTemplate(promptTemplate, promptVars)

	resp, err := model.Call(ctx, prompt, nil)
	if err != nil {
		return "", err
	}
	greeting := strings.TrimSpace(resp.TextContent)
	if greeting == "" {
		return "", fmt.Errorf("llm returned empty greeting")
	}
	return greeting, nil
}

// greetingLLM loads the LLM plugin configured for the greeting hook. The plugin
// name is taken from "llm_type" (default OpenAILLM) and its configuration from
// the optional "llm_config" map, with "llm_model" provided as a convenience
// override for the model field.
func (r *Runner) greetingLLM(cfg map[string]any) (llm.LLM, error) {
	llmType := stringVal(cfg, "llm_type")
	if llmType == "" {
		llmType = defaultGreetingLLMType
	}

	llmCfg := map[string]any{}
	if apiKey := stringVal(cfg, "api_key"); apiKey != "" {
		llmCfg["api_key"] = apiKey
	}
	if nested, ok := cfg["llm_config"].(map[string]any); ok {
		for k, v := range nested {
			llmCfg[k] = v
		}
	}
	if model := stringVal(cfg, "llm_model"); model != "" {
		llmCfg["model"] = model
	}

	return llm.Load(llmType, llmCfg)
}

// staticGreeting reproduces the original, deterministic greeting used as a
// fallback when LLM generation is unavailable.
func staticGreeting(snapshot providers.PresenceSnapshot, snapErr error, robotName, helpMessage string) string {
	robotIntro := ""
	if robotName != "" {
		robotIntro = fmt.Sprintf("I'm %s. ", robotName)
	}

	switch {
	case snapErr != nil:
		return fmt.Sprintf("Hello! %s%s", robotIntro, helpMessage)
	case !strings.EqualFold(snapshot.ClosestName, "unknown"):
		return fmt.Sprintf("Hello %s! %sNice to see you. %s", snapshot.ClosestName, robotIntro, helpMessage)
	default:
		return fmt.Sprintf("Hello! %s%s", robotIntro, helpMessage)
	}
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
