package tracetype

type TraceRecord struct {
	Timestamp  string           `json:"ts"`
	Generation int              `json:"generation"`
	LLMInput   string           `json:"llm_input"`
	LLMOutput  []map[string]any `json:"llm_output"`
}
