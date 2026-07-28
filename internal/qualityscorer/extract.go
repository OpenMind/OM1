package qualityscorer

import (
	"regexp"
	"strings"
)

// voiceRe matches OM1's `Voice: "..."` marker; two quote alternatives since Go's RE2 has no backreferences.
var voiceRe = regexp.MustCompile(`(?s)Voice:\s*(?:"([^"]*)"|'([^']*)')`)

// extractPrompt returns the first Voice: marker's contents from llmInput, or "" if none.
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

// extractResponse returns the spoken reply from an LLMOutput action list; empty means the robot said nothing that turn.
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
