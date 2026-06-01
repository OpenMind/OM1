package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
	"time"

	"github.com/openmind/om1/internal/llm"
	"github.com/openmind/om1/internal/logger"
	"go.uber.org/zap"
)

// cloneStringAnyMap returns a shallow copy so a composite can inject parent
// settings (e.g. api_key) into a sub-LLM config without mutating the source map.
func cloneStringAnyMap(m map[string]any) map[string]any {
	clone := make(map[string]any, len(m)+1)
	for k, v := range m {
		clone[k] = v
	}
	return clone
}

func init() {
	llm.Register("DualLLM", NewDual)
}

const (
	// dualTimeout is the window within which a sub-LLM response is considered "in time".
	dualTimeout = 3200 * time.Millisecond
	// dualEvalTimeout bounds the quality-evaluation call when both models respond.
	dualEvalTimeout = 2 * time.Second

	dualEvalBaseURL = "http://127.0.0.1:8860/v1"
	dualEvalAPIKey  = "local"

	defaultDualLocalType  = "QwenLLM"
	defaultDualLocalModel = "RedHatAI/Qwen3-30B-A3B-quantized.w4a16"
	defaultDualCloudType  = "OpenAILLM"
	defaultDualCloudModel = "gpt-4.1"
)

// voiceInputRe extracts the user's voice input from a fused prompt for eval context.
var voiceInputRe = regexp.MustCompile(`Voice:\s*([^\n]+)`)

type dualConfig struct {
	LocalType   string         `json:"local_llm_type"`
	LocalConfig map[string]any `json:"local_llm_config"`
	CloudType   string         `json:"cloud_llm_type"`
	CloudConfig map[string]any `json:"cloud_llm_config"`
	APIKey      string         `json:"api_key"`
}

// dualLLM races a local and a cloud sub-LLM and selects between them:
//  1. both respond in time → prefer the one with function calls, else evaluate quality;
//  2. one responds in time → use it;
//  3. neither in time → use whichever completes first.
type dualLLM struct {
	local llm.LLM
	cloud llm.LLM
	eval  *openAICompatLLM
}

// NewDual creates a DualLLM that wraps a local and cloud sub-LLM from the registry.
func NewDual(configMap map[string]any) (llm.LLM, error) {
	var cfg dualConfig
	if err := remarshal(configMap, &cfg); err != nil {
		return nil, fmt.Errorf("DualLLM config: %w", err)
	}

	if cfg.LocalType == "" {
		cfg.LocalType = defaultDualLocalType
	}
	if cfg.CloudType == "" {
		cfg.CloudType = defaultDualCloudType
	}
	localCfg := cloneStringAnyMap(cfg.LocalConfig)
	if _, ok := localCfg["model"]; !ok {
		localCfg["model"] = defaultDualLocalModel
	}
	cloudCfg := cloneStringAnyMap(cfg.CloudConfig)
	if _, ok := cloudCfg["model"]; !ok {
		cloudCfg["model"] = defaultDualCloudModel
	}
	// The cloud model needs the agent's API key; the local model runs unauthenticated.
	if cfg.APIKey != "" {
		cloudCfg["api_key"] = cfg.APIKey
	}

	local, err := llm.Load(cfg.LocalType, localCfg)
	if err != nil {
		return nil, fmt.Errorf("DualLLM: load local %q: %w", cfg.LocalType, err)
	}
	cloud, err := llm.Load(cfg.CloudType, cloudCfg)
	if err != nil {
		return nil, fmt.Errorf("DualLLM: load cloud %q: %w", cfg.CloudType, err)
	}

	evalModel := defaultDualLocalModel
	if m, ok := localCfg["model"].(string); ok && m != "" {
		evalModel = m
	}
	eval := &openAICompatLLM{
		provider: "DualLLM-eval",
		config:   compatConfig{APIKey: dualEvalAPIKey, Model: evalModel, BaseURL: dualEvalBaseURL},
		extraBody: map[string]any{
			"temperature":          0.0,
			"max_tokens":           10,
			"chat_template_kwargs": map[string]any{"enable_thinking": false},
		},
	}

	return &dualLLM{local: local, cloud: cloud, eval: eval}, nil
}

// SetSchemas propagates the tool schemas to both sub-LLMs.
func (d *dualLLM) SetSchemas(schemas []map[string]any) {
	d.local.SetSchemas(schemas)
	d.cloud.SetSchemas(schemas)
}

// FunctionSchemas returns the schemas configured on the local sub-LLM.
func (d *dualLLM) FunctionSchemas() []map[string]any { return d.local.FunctionSchemas() }

// dualResult carries a sub-LLM response along with which source produced it and how long it took.
type dualResult struct {
	resp    *llm.Response
	elapsed time.Duration
	source  string
}

// Call races the two sub-LLMs and returns the selected response.
func (d *dualLLM) Call(ctx context.Context, prompt string, history []llm.Message) (*llm.Response, error) {
	start := time.Now()
	voiceInput := extractVoiceInput(prompt)

	callCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	results := make(chan dualResult, 2)
	run := func(sub llm.LLM, source string) {
		callStart := time.Now()
		resp, err := sub.Call(callCtx, prompt, history)
		if err != nil {
			logger.Get().Warn("DualLLM sub-call failed", zap.String("source", source), zap.Error(err))
			resp = nil
		}
		results <- dualResult{resp: resp, elapsed: time.Since(callStart), source: source}
	}
	go run(d.local, "local")
	go run(d.cloud, "cloud")

	// Collect responses until both arrive or the in-time window elapses.
	var arrived []dualResult
	deadline := time.NewTimer(dualTimeout)
	defer deadline.Stop()
collect:
	for len(arrived) < 2 {
		select {
		case r := <-results:
			arrived = append(arrived, r)
		case <-deadline.C:
			break collect
		}
	}

	inTime := map[string]dualResult{}
	for _, r := range arrived {
		if r.resp != nil && r.elapsed <= dualTimeout {
			inTime[r.source] = r
		}
	}

	var chosen *llm.Response
	switch len(inTime) {
	case 2:
		chosen = d.selectBest(callCtx, inTime["local"].resp, inTime["cloud"].resp, voiceInput)
	case 1:
		for _, r := range inTime {
			chosen = r.resp
		}
		cancel() // stop the slower sub-LLM
	default:
		// Neither in time: use the first usable response, waiting if none has arrived yet.
		chosen = firstUsable(arrived)
		for chosen == nil && len(arrived) < 2 {
			r := <-results
			arrived = append(arrived, r)
			if r.resp != nil {
				chosen = r.resp
			}
		}
		cancel()
	}

	logger.Get().Info("DualLLM",
		zap.Int("in_time", len(inTime)),
		zap.Int64("elapsed_ms", time.Since(start).Milliseconds()),
	)

	if chosen == nil {
		return &llm.Response{}, nil
	}
	return chosen, nil
}

// selectBest applies the dual selection rules when both sub-LLMs responded in time.
func (d *dualLLM) selectBest(ctx context.Context, local, cloud *llm.Response, voiceInput string) *llm.Response {
	localHasCalls := len(local.ToolCalls) > 0
	cloudHasCalls := len(cloud.ToolCalls) > 0

	switch {
	case localHasCalls && !cloudHasCalls:
		return local
	case cloudHasCalls && !localHasCalls:
		return cloud
	case !localHasCalls && !cloudHasCalls:
		return local
	}

	if d.evaluateQuality(ctx, local, cloud, voiceInput) == "cloud" {
		return cloud
	}
	return local
}

// evaluateQuality asks the local eval model which response better answers the
// user. It returns "local" or "cloud", defaulting to "local" on any failure.
func (d *dualLLM) evaluateQuality(ctx context.Context, local, cloud *llm.Response, prompt string) string {
	localActions, _ := json.MarshalIndent(toolCallsToEval(local.ToolCalls), "", "  ")
	cloudActions, _ := json.MarshalIndent(toolCallsToEval(cloud.ToolCalls), "", "  ")
	if len(prompt) > 500 {
		prompt = prompt[:500]
	}

	evalPrompt := fmt.Sprintf(`You are evaluating two AI responses to determine which better answers the user's question.

Original User Question/Context:
%s

Response A (local model):
%s

Response B (cloud model):
%s

Evaluate based on:
1. Relevance - Which response better addresses the user's question?
2. Completeness - Does it fully answer what was asked?
3. Appropriateness - Are the actions suitable for the context?
4. Quality - Is the content natural and engaging?

Respond with ONLY a single word: either "A" or "B" for the better response.`, prompt, localActions, cloudActions)

	evalCtx, cancel := context.WithTimeout(ctx, dualEvalTimeout)
	defer cancel()

	resp, err := d.eval.Call(evalCtx, evalPrompt, nil)
	logger.Get().Info("DualLLM eval", zap.String("response", resp.TextContent), zap.Error(err))
	if err != nil || resp == nil {
		logger.Get().Warn("DualLLM quality evaluation failed, defaulting to local", zap.Error(err))
		return "local"
	}
	if strings.Contains(strings.ToUpper(strings.TrimSpace(resp.TextContent)), "A") {
		return "local"
	}
	return "cloud"
}

// toolCallsToEval renders tool calls as {"type","value"} entries for the eval prompt.
func toolCallsToEval(calls []llm.ToolCall) []map[string]any {
	out := make([]map[string]any, 0, len(calls))
	for _, tc := range calls {
		out = append(out, map[string]any{"type": tc.Name, "value": tc.Arguments})
	}
	return out
}

// firstUsable returns the first non-nil response in arrival order, or nil.
func firstUsable(results []dualResult) *llm.Response {
	for _, r := range results {
		if r.resp != nil {
			return r.resp
		}
	}
	return nil
}

// extractVoiceInput pulls the "Voice: ..." line out of a fused prompt.
func extractVoiceInput(prompt string) string {
	if m := voiceInputRe.FindStringSubmatch(prompt); m != nil {
		return strings.TrimSpace(m[1])
	}
	return ""
}
