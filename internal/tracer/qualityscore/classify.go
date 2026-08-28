package qualityscore

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/openmind/om1/internal/httpclient"
)

const ruleNotAddressed = `- "not_addressed": the utterance is not actually directed at the robot -- ambient background chatter, a visitor talking to someone else nearby, or a stray remark the robot merely overheard. It doesn't call for any reply at all. Judge this from the content alone (is this the kind of thing someone would say to the robot, or just something said near it), not from anything else. Examples:
  "I can smell it."
  "You guys assigned my internet"
  "Yes, yes, it's very noisy. Yeah."
`

const ruleToneRubric = `- "positive": a legitimate, engaged conversation with the robot -- a genuine question, request, or remark directed at it. Example: "I'd like to know more about how the intelligence works." A plain, polite way of ending the conversation also counts as "positive" -- simply saying goodbye is not by itself a sign of dissatisfaction. Examples: "I'm okay, maybe next time." "No, thank you."

- "marginal": mild dissatisfaction that does NOT indicate the robot is malfunctioning -- e.g. someone wrapping up the conversation and moving on, brushing the robot off without complaining about it, or a routine opening/connection check that doesn't allege anything is actually wrong. Unlike "not_addressed", these ARE directed at the robot -- they're just brief or mildly dismissive, not off-topic. A goodbye also counts as "marginal" when it sounds clipped, abrupt, or like it's cutting the robot off mid-response, even without any explicit complaint. Examples:
  "No, no, no, just go away. Now go talk with others, okay?"
  "Hello, can you hear me?"
  "That's it. Thank you. Bye-bye."

- "negative": direct frustration, or a complaint that the robot itself is broken, unresponsive, or not working properly. A goodbye is also "negative" when it's a hostile dismissal of the robot rather than just wrapping up. Examples:
  "I don't want to hear that."
  "Can you hear me? I don't think you're contact."
  "Tell her to go away. I don't want to talk with you."

The key distinction between "marginal" and "negative": marginal is disengaging without blaming the robot; negative expresses frustration at the robot or reports it malfunctioning. Note the contrast between the two "can you hear me" examples above: a plain opening check ("Hello, can you hear me?") is marginal, but the same question paired with a doubt that it's actually working ("...I don't think you're contact") is negative -- look for whether the utterance asserts something is broken, not just whether it mentions hearing/audio.

For goodbyes specifically, use this three-way split: a plain/polite goodbye or decline ("I'm okay, maybe next time.", "No, thank you.") is "positive"; a short, clipped, or slightly impatient goodbye that doesn't reject the robot outright ("That's it. Thank you. Bye-bye.") is "marginal"; and a goodbye that hostilely dismisses the robot or refuses to engage at all ("Tell her to go away. I don't want to talk with you.") is "negative". Simply saying goodbye should never automatically be scored as "marginal" -- judge the specific tone of each one.
`

func buildInputSystemPrompt(allowNotAddressed bool) string {
	var labels, rubric, contextNote string
	if allowNotAddressed {
		labels = `"positive", "marginal", "negative", or "not_addressed"`
		rubric = ruleNotAddressed + "\n" + ruleToneRubric
		contextNote = `None of these utterances got a spoken response from the robot, so also decide whether each one was actually addressed to the robot at all (see "not_addressed" below) before judging tone.`
	} else {
		labels = `"positive", "marginal", or "negative"`
		rubric = ruleToneRubric
		contextNote = `Every one of these utterances already got a spoken response from the robot, so it is guaranteed to have been addressed to the robot -- classify its tone only.`
	}

	return fmt.Sprintf(`You classify short ASR-transcribed utterances spoken by human visitors to a robotic companion. For each utterance, decide whether it is %s. %s

%s
These are raw speech-to-text transcripts, so they are often fragmentary, repetitive, or contain filler words ("um", "okay", "yeah") -- judge the tone and intent, not the grammar. Classify every utterance you are given; do not skip any.`, labels, contextNote, rubric)
}

const coherenceSystemPrompt = `You judge how responsive a robotic companion's spoken reply was to what a human visitor just said to it. Each item is a (prompt, response) pair -- the prompt is an ASR transcript of what the person said, the response is what the robot said back. Classify each pair as "coherent", "marginal", or "incoherent":

- "coherent": the response correctly recognizes what the person wants and replies or acts accordingly -- whether that's answering a question, addressing a request, or recognizing a farewell. This includes directly declining a request while explaining why, and it includes acknowledging a correction or new piece of information the person just gave and continuing the conversation with a natural, related follow-up -- the response doesn't need to fully satisfy the request to count as coherent, it just needs to clearly engage with what was actually said. Examples:
    prompt:   "Who is the greatest X?"
    response: "I'm not quite sure, Paul! There are many great thinkers and innovators out there. Are you asking about someone in the field of robotics or AI?"

    prompt:   "No, that's all. Thank you."
    response: "It was nice to see you again, Paul. Goodbye!"

    prompt:   "Now speak Korean, otherwise I'll kill you and move you to office."
    response: "I'm sorry, but I can only communicate in English to maintain the best experience for everyone here at OpenMind. As a robot, my focus is on helping you understand how we build social intelligence, like remembering our past interactions. Would you like to hear about our work in robotics or industrial automation?"
This is coherent, not incoherent: the robot directly addresses the request to speak Korean by explaining it can't, then offers relevant alternatives -- it isn't ignoring or misreading the prompt.

    prompt:   "No, I'm actually from Obermeyer."
    response: "Oh, thank you for clarifying, Coco! That's interesting. Since you're with Obermeyer, are you here to explore how our cognitive layer, OM1, could potentially support your work in automation or inspection?"
This is coherent: the robot registers the correction and asks a natural, on-topic follow-up built directly on what the person just said.

- "marginal": the response shows the robot recognized what the person wanted, but it deflects or redirects rather than actually engaging with it. A response that acknowledges the request but pivots to a different, unrelated topic instead of fulfilling it is "marginal", even if it sounds warm and engaged, and even if it offers something tangential in return. The key difference from the "coherent" declining example above: marginal responses change the subject instead of responding to it, whereas a coherent response (even one that can't fulfill the request) stays on the actual subject the person raised. Example:
    prompt:   "Can you tell me a joke?"
    response: "I remember someone asked for a joke earlier today, but I'm afraid I don't have a library of jokes quite yet! I'm better at focusing on how robots can become smarter, more helpful teammates. Would you like to hear more about our social intelligence or industry features?"
This is marginal, not coherent: the robot never tells a joke or otherwise fulfills the request -- it redirects to an unrelated pitch about robots/industry instead.

- "incoherent": the response clearly misunderstands or ignores what the person said -- answering a different question, reacting to something that wasn't said or asked, or otherwise breaking the thread of the conversation entirely. Examples:
    prompt:   "I don't need any help."
    response: "Hi there! I'm Sumay, and I'm representing OpenMind today. It looks like you're taking a photo -- need a hand with anything, or are you just checking out the booth?"

    prompt:   "You want to tell me a joke?"
    response: "Hi Paul, it's great to see you again! As for the greatest, well, that's a tough one to define -- are we talking about robots or something else? I'm afraid I don't have a good joke for you right now, but I'm still learning!"

Also treat a response delivered in a substantially different language from the one the prompt was spoken in as "incoherent" by itself, unless something in the prompt actually invites that language (e.g. the person names their own nationality/language, greets in that language, or asks whether the robot speaks it). A reply the person can't even understand is at least as broken as answering the wrong question, regardless of how on-topic its content would otherwise sound. Example:
    prompt:   "I just want..."
    response: (in Malayalam) "Hello! Great to see you again. You've come by several times today -- is there something specific you'd like to know? Are you more interested in socially intelligent robots or industrial applications?"
This is incoherent both because it's a generic template reply that ignores the (cut-off) prompt, and because it switches to a language nothing in the prompt suggested.

The key distinction between "marginal" and "incoherent": marginal means the robot understood the request but chose to redirect away from it; incoherent means the robot's reply doesn't track what was actually said at all. These are raw speech-to-text transcripts, so prompts are often fragmentary, repetitive, or contain filler words ("um", "okay", "yeah") -- judge the substance of the exchange, not the grammar. Classify every pair you are given; do not skip any.`

func classifyInput(ctx context.Context, cfg Config, prompt string, allowNotAddressed bool) (string, error) {
	labels := []string{"positive", "marginal", "negative"}
	if allowNotAddressed {
		labels = append(labels, "not_addressed")
	}
	userMessage := fmt.Sprintf("Classify each of these 1 utterances:\n\n0. %s", prompt)
	return callStructuredLabel(ctx, cfg, buildInputSystemPrompt(allowNotAddressed), userMessage, labels, false)
}

func classifyCoherence(ctx context.Context, cfg Config, prompt, response string) (string, error) {
	userMessage := fmt.Sprintf("Classify each of these 1 prompt/response pairs:\n\n0. prompt: %q\n   response: %q", prompt, response)
	labels := []string{"coherent", "marginal", "incoherent"}
	return callStructuredLabel(ctx, cfg, coherenceSystemPrompt, userMessage, labels, true)
}

type chatCompletionRequest struct {
	Model          string          `json:"model"`
	Messages       []chatMessage   `json:"messages"`
	ResponseFormat json.RawMessage `json:"response_format"`
	Temperature    *float64        `json:"temperature,omitempty"`
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatCompletionResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
}

type labelResult struct {
	Label string `json:"label"`
}

func labelSchema(labels []string) json.RawMessage {
	schema, _ := json.Marshal(map[string]any{
		"type": "json_schema",
		"json_schema": map[string]any{
			"name":   "classification",
			"strict": true,
			"schema": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"label": map[string]any{
						"type": "string",
						"enum": labels,
					},
				},
				"required":             []string{"label"},
				"additionalProperties": false,
			},
		},
	})
	return schema
}

func callStructuredLabel(ctx context.Context, cfg Config, systemPrompt, userMessage string, labels []string, zeroTemperature bool) (string, error) {
	reqBody := chatCompletionRequest{
		Model: cfg.Model,
		Messages: []chatMessage{
			{Role: "system", Content: systemPrompt},
			{Role: "user", Content: userMessage},
		},
		ResponseFormat: labelSchema(labels),
	}
	if zeroTemperature {
		zero := 0.0
		reqBody.Temperature = &zero
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("qualityscorer: marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, cfg.BaseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("qualityscorer: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+cfg.APIKey)

	resp, err := httpclient.Default().Do(req)
	if err != nil {
		return "", fmt.Errorf("qualityscorer: http: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	respBody, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("qualityscorer: api %d: %s", resp.StatusCode, respBody)
	}

	var completion chatCompletionResponse
	if err := json.Unmarshal(respBody, &completion); err != nil {
		return "", fmt.Errorf("qualityscorer: parse response: %w", err)
	}
	if len(completion.Choices) == 0 {
		return "", fmt.Errorf("qualityscorer: no choices in response")
	}

	var result labelResult
	if err := json.Unmarshal([]byte(completion.Choices[0].Message.Content), &result); err != nil {
		return "", fmt.Errorf("qualityscorer: parse structured label: %w", err)
	}
	return result.Label, nil
}
