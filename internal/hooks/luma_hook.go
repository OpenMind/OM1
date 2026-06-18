package hooks

import (
	"context"
)

func init() {
	RegisterHook("luma_hook", "luma_intro_hook", (*Runner).lumaIntroHook)
}

const defaultLumaHelp = "Are you registered on Luma?"

const defaultLumaPrompt = "You are {robot_name}, a friendly robot welcoming a guest to an event. " +
	"The current time is {current_time}. " +
	"Generate a single warm, natural spoken greeting of one or two short sentences. " +
	"Here is what you currently see: {scene}. " +
	"Make it feel personal and present by naturally referencing something specific from what you see; never invent anything. " +
	"If a specific person is recognized ({closest_name}), greet them by name; otherwise greet generically. " +
	"{memory}" +
	"Finish by asking the guest whether they are registered on Luma, for example: \"{help_message}\". " +
	"Respond with only the greeting text, with no quotes or commentary."

func (r *Runner) lumaIntroHook(ctx context.Context, cfg, vars map[string]any) error {
	return r.announceGenerated(ctx, cfg, vars, defaultLumaPrompt, defaultLumaHelp)
}
