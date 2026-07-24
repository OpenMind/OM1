package qualityscorer

import (
	"regexp"
	"strings"
)

// voiceRe extracts the ASR transcript embedded in OM1's constructed LLM
// prompt via a literal `Voice: "..."` (or `Voice: '...'`) marker. Ported from
// dataAnalysis's extract_prompt_response_pairs.py VOICE_RE
// (r'Voice:\s*(["\'])(.*?)\1', re.S); Go's RE2 has no backreferences, so the
// two quote styles are two alternatives instead of one backreferenced group
// -- semantically equivalent for this pattern.
var voiceRe = regexp.MustCompile(`(?s)Voice:\s*(?:"([^"]*)"|'([^']*)')`)

// extractPrompt returns the ASR transcript embedded in llmInput, or "" if no
// Voice: marker is present. Only the first match is used, matching the
// Python original (only the latest heard utterance, not full chat history).
func extractPrompt(llmInput string) string {
	loc := voiceRe.FindStringSubmatchIndex(llmInput)
	if loc == nil {
		return ""
	}
	if loc[2] != -1 { // double-quoted group participated
		return strings.TrimSpace(llmInput[loc[2]:loc[3]])
	}
	if loc[4] != -1 { // single-quoted group participated
		return strings.TrimSpace(llmInput[loc[4]:loc[5]])
	}
	return ""
}

// extractResponse returns (responseText, responseType) from a trace record's
// LLMOutput action list, ported from extract_prompt_response_pairs.py's
// extract_response. Observed action types: {"type": "greeting_conversation",
// "value": {"response": "..."}} and {"type": "speak", "value": {"action":
// "..."}} carry spoken replies; other types (robot_action, emotion,
// face_memory, ...) carry no spoken text and are skipped. An empty response
// is the correct, intentional signal that the robot said nothing that turn --
// not a parse failure.
func extractResponse(llmOutput []map[string]any) (response string, responseType string) {
	for _, item := range llmOutput {
		value, ok := item["value"].(map[string]any)
		if !ok {
			continue
		}
		itemType, _ := item["type"].(string)
		switch itemType {
		case "greeting_conversation":
			if resp, ok := value["response"].(string); ok {
				return strings.TrimSpace(resp), "greeting_conversation"
			}
		case "speak":
			if action, ok := value["action"].(string); ok {
				return strings.TrimSpace(action), "speak"
			}
		}
	}
	return "", ""
}
