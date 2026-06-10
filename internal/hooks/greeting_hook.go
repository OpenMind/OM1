package hooks

import (
	"context"
	"encoding/base64"
	"fmt"
	"strings"
	"time"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/llm"
	"github.com/openmind/om1/internal/providers"
	"github.com/openmind/om1/internal/providers/tts"
	video "github.com/openmind/om1/internal/providers/vlm"
	"github.com/openmind/om1/internal/util"
)

func init() {
	RegisterHook("greeting_hook", "greeting_start_hook", (*Runner).greetingStartHook)
	RegisterHook("greeting_hook", "greeting_end_hook", (*Runner).greetingEndHook)
}

const defaultGreetingLLMType = "GeminiLLM"

const (
	defaultVisionBaseURL     = "https://api.openmind.com/api/core/gemini"
	defaultVisionModel       = "gemini-2.5-flash"
	defaultVisionMaxTok      = 1024
	defaultVisionRTSPURL     = "rtsp://localhost:8554/top_camera_raw"
	defaultVisionGrabTimeout = 2 * time.Second
)

const defaultGreetingPrompt = "You are {robot_name}, a friendly robot greeting whoever is in front of you. " +
	"The current time is {current_time}. " +
	"Generate a single warm, natural spoken greeting of one or two short sentences. " +
	"Here is what you currently see: {scene}. " +
	"If a specific person is recognized ({closest_name}), greet them by name; otherwise greet generically. " +
	"You may reflect the time of day in your greeting when it feels natural. " +
	"Finish by offering help, for example: \"{help_message}\". " +
	"Respond with only the greeting text, with no quotes or commentary."

const defaultVisionGreetingPrompt = "You are {robot_name}, a friendly robot greeting whoever is right in front of you. " +
	"The current time is {current_time}. " +
	"Look carefully at the attached image from your camera and generate a single warm, natural spoken greeting of one or two short sentences. " +
	"Make it feel personal and present by naturally referencing something specific you genuinely see — for example the person's appearance or clothing, what they are doing or holding, how many people are present, or the setting around them. " +
	"Only mention details you can actually see in the image; never invent anything. " +
	"If you recognize a specific person ({closest_name}), greet them by name; otherwise greet generically. " +
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

// generateGreeting constructs the prompt for the greeting, attempts to generate a greeting using vision when possible, and falls back to text-only LLM generation when vision fails or is unavailable.
// ToDo:
// - add more context to the prompt, e.g. recent conversation history if available, etc.
func (r *Runner) generateGreeting(ctx context.Context, cfg, vars map[string]any, snapshot providers.PresenceSnapshot, robotName, helpMessage string) (string, error) {
	if robotName == "" {
		robotName = "a friendly robot"
	}

	scene := strings.TrimSpace(snapshot.ToText())
	if scene == "" {
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

	visionTemplate := stringVal(cfg, "vision_prompt")
	if strings.TrimSpace(visionTemplate) == "" {
		visionTemplate = defaultVisionGreetingPrompt
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

	if greeting, ok := r.visionGreeting(ctx, cfg, formatTemplate(visionTemplate, promptVars)); ok {
		return greeting, nil
	}

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

// visionGreeting attempts to generate a greeting using the vision describer. It returns the greeting and a boolean indicating whether generation was successful.
func (r *Runner) visionGreeting(ctx context.Context, cfg map[string]any, prompt string) (string, bool) {
	apiKey := stringVal(cfg, "api_key")
	if apiKey == "" {
		return "", false
	}

	grabTimeout := defaultVisionGrabTimeout
	if sec := util.FloatFrom(cfg["vlm_grab_timeout_sec"], 0); sec > 0 {
		grabTimeout = time.Duration(sec * float64(time.Second))
	}

	grabCtx, cancel := context.WithTimeout(ctx, grabTimeout)
	defer cancel()

	jpeg, err := video.GrabFrame(grabCtx, video.VideoRTSPStreamConfig{
		RTSPURL: util.FirstNonEmpty(stringVal(cfg, "rtsp_url"), defaultVisionRTSPURL),
	})
	if err != nil {
		r.log.Warn("greeting_start_hook: failed to grab frame, falling back to text-only greeting", zap.Error(err))
		return "", false
	}
	encoded := base64.StdEncoding.EncodeToString(jpeg)

	describer := video.NewDescriber(video.Describer{
		Name:      "greeting_hook",
		APIKey:    apiKey,
		BaseURL:   util.FirstNonEmpty(stringVal(cfg, "base_url"), defaultVisionBaseURL),
		Model:     util.FirstNonEmpty(stringVal(cfg, "model"), defaultVisionModel),
		Prompt:    prompt,
		MaxTokens: int(util.FloatFrom(cfg["max_tokens"], defaultVisionMaxTok)),
		Log:       r.log,
	})

	greeting, err := describer.Describe(ctx, encoded)
	if err != nil {
		r.log.Warn("greeting_start_hook: vision greeting failed, falling back to text", zap.Error(err))
		return "", false
	}
	greeting = strings.TrimSpace(greeting)
	if greeting == "" {
		return "", false
	}
	return greeting, true
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
