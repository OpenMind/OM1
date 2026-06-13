package hooks

import (
	"context"
	"fmt"
	"strings"
	"time"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/llm"
	"github.com/openmind/om1/internal/providers"
	"github.com/openmind/om1/internal/providers/tts"
	video "github.com/openmind/om1/internal/providers/vlm"
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
	"Make it feel personal and present by naturally referencing something specific from what you see; never invent anything. " +
	"If a specific person is recognized ({closest_name}), greet them by name; otherwise greet generically. " +
	"{memory}" +
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

	memContext := r.recallMemory(ctx, snapshot.ClosestUUID)

	if greeting, genErr := r.generateGreeting(ctx, cfg, vars, snapshot, memContext, robotName, helpMessage); genErr != nil {
		r.log.Warn("greeting_start_hook: llm generation failed, using static greeting", zap.Error(genErr))
		provider.AddText(staticGreeting(snapshot, snapErr, robotName, helpMessage))
	} else {
		r.log.Info("greeting generated successfully", zap.String("greeting", greeting))
		provider.AddText(greeting)
	}

	r.log.Info("greeting start hook executed successfully")
	return nil
}

// recallMemory attempts to recall relevant memory context for the user identified by uuid,
// returning the context string or "" if memory is unavailable or an error occurs.
func (r *Runner) recallMemory(ctx context.Context, uuid string) string {
	if r.memory == nil || uuid == "" {
		return ""
	}

	memContext := strings.TrimSpace(r.memory.SearchAndFormat(ctx, "", uuid))
	if memContext != "" {
		r.log.Info("greeting_start_hook: recalled memory",
			zap.String("uuid", uuid), zap.Int("chars", len(memContext)))
	}
	return memContext
}

func memoryClause(memContext string) string {
	if memContext == "" {
		return ""
	}
	return "Here is what you remember about them from past interactions: " + memContext +
		" Naturally reference this to make the greeting personal, but never invent details. "
}

// generateGreeting constructs a prompt using the snapshot and other context, calls the LLM to generate a greeting, and returns the greeting text.
func (r *Runner) generateGreeting(ctx context.Context, cfg, vars map[string]any, snapshot providers.PresenceSnapshot, memContext, robotName, helpMessage string) (string, error) {
	if robotName == "" {
		robotName = "a friendly robot"
	}

	scene := r.sceneDescription(snapshot)

	closestName := snapshot.ClosestName
	if closestName == "" || strings.EqualFold(closestName, "unknown") {
		closestName = "no one in particular"
	}

	promptTemplate := stringVal(cfg, "prompt")
	if strings.TrimSpace(promptTemplate) == "" {
		promptTemplate = defaultGreetingPrompt
	}

	promptVars := make(map[string]any, len(vars)+6)
	for k, v := range vars {
		promptVars[k] = v
	}
	promptVars["robot_name"] = robotName
	promptVars["scene"] = scene
	promptVars["closest_name"] = closestName
	promptVars["help_message"] = helpMessage
	promptVars["current_time"] = time.Now().Format("Monday, January 2, 2006 at 3:04 PM")
	promptVars["memory"] = memoryClause(memContext)

	prompt := formatTemplate(promptTemplate, promptVars)

	model, err := r.greetingLLM(cfg)
	if err != nil {
		return "", err
	}

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

// sceneDescription constructs a natural-language description of the current scene based on the presence snapshot and latest VLM description.
func (r *Runner) sceneDescription(snapshot providers.PresenceSnapshot) string {
	parts := make([]string, 0, 2)
	if faces := strings.TrimSpace(snapshot.ToText()); faces != "" {
		parts = append(parts, faces)
	}

	if desc, _, ok := video.LatestDescription().Get(); ok {
		if desc = strings.TrimSpace(desc); desc != "" {
			parts = append(parts, desc)
		}
	}

	if len(parts) == 0 {
		return "No one is clearly in view."
	}

	return strings.Join(parts, " ")
}

// greetingLLM resolves the LLM for a greeting hook, applying configuration from both the top-level and nested "llm_config" keys.
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
func (r *Runner) greetingTTSProvider(cfg map[string]any) (*tts.ElevenLabsProvider, error) {
	ttsName := strings.ToLower(stringVal(cfg, "tts_provider"))

	if ttsName != "" && ttsName != "elevenlabs" {
		r.log.Warn("greeting hook: TTS provider not supported, falling back to elevenlabs",
			zap.String("tts_provider", ttsName))
	}

	return tts.ElevenLabs(elevenLabsConfigFrom(cfg), r.log), nil
}
