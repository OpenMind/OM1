package fuser

import (
	"context"
	"strings"

	"github.com/openmind/om1/internal/actions"
	"github.com/openmind/om1/internal/config"
	"github.com/openmind/om1/internal/knowledgebase"
	"github.com/openmind/om1/internal/logger"
	"github.com/openmind/om1/internal/providers"
)

type Fuser struct {
	runtimeConfig *config.RuntimeConfig
	agentActions  []*actions.AgentAction
	knowledgeBase *knowledgebase.KnowledgeBase
	kbMinScore    float64
}

func NewFuser(runtimeConfig *config.RuntimeConfig, agentActions []*actions.AgentAction) *Fuser {
	f := &Fuser{runtimeConfig: runtimeConfig, agentActions: agentActions}

	if runtimeConfig.KnowledgeBase != nil {
		log := logger.Get()
		kb, err := knowledgebase.New(runtimeConfig.KnowledgeBase, log)
		if err != nil {
			log.Warn("failed to initialize knowledge base, continuing without RAG")
		} else if kb != nil {
			f.knowledgeBase = kb
			f.kbMinScore = runtimeConfig.KnowledgeBase.MinScore
		}
	}

	return f
}

// Fuse combines the prompt components into a single string to be sent to the LLM.
func (f *Fuser) Fuse(ctx context.Context, sensorBuffers []string) (string, error) {
	var builder strings.Builder

	// System prompt base + governance.
	builder.WriteString(f.runtimeConfig.SystemPromptBase)
	builder.WriteString("\n\n")
	if f.runtimeConfig.SystemGovernance != "" {
		builder.WriteString("Governance rules:\n")
		builder.WriteString(f.runtimeConfig.SystemGovernance)
		builder.WriteString("\n\n")
	}

	// 2. Sensor inputs.
	hasObservations := false
	for _, buffer := range sensorBuffers {
		if buffer != "" {
			if !hasObservations {
				builder.WriteString("Current observations:\n")
				hasObservations = true
			}
			builder.WriteString("- ")
			builder.WriteString(buffer)
			builder.WriteString("\n")
		}
	}
	if hasObservations {
		builder.WriteString("\n")
	}

	// 3. Knowledge-base.
	if f.knowledgeBase != nil {
		if question := f.voiceQuery(); question != "" {
			documents, err := f.knowledgeBase.Query(ctx, question, f.runtimeConfig.KnowledgeBase.TopK)
			if err == nil && len(documents) > 0 {
				builder.WriteString("Relevant context:\n")
				for _, document := range documents {
					builder.WriteString("- ")
					builder.WriteString(document)
					builder.WriteString("\n")
				}
				builder.WriteString("\n")
			}
		}
	}

	// 4. Available actions description.
	visibleActions := f.visibleActions()
	if len(visibleActions) > 0 {
		builder.WriteString("Available actions:\n")
		for _, action := range visibleActions {
			builder.WriteString("- ")
			builder.WriteString(action.LLMLabel)
			builder.WriteString("\n")
		}
		builder.WriteString("\n")
	}

	// 5. Examples.
	if f.runtimeConfig.PromptExamples != "" {
		builder.WriteString(f.runtimeConfig.PromptExamples)
		builder.WriteString("\n\n")
	}

	// 6. Closing question.
	builder.WriteString("What will you do next?")

	return builder.String(), nil
}

// visibleActions filters the agent actions to include only those that are not marked as ExcludeFromPrompt, meaning they should be included in the prompt for the LLM.
func (f *Fuser) visibleActions() []*actions.AgentAction {
	var visible []*actions.AgentAction
	for _, action := range f.agentActions {
		if !action.ExcludeFromPrompt {
			visible = append(visible, action)
		}
	}
	return visible
}

// buildQuestion constructs a question to query the knowledge base.
func (f *Fuser) voiceQuery() string {
	io := providers.IO()
	voice := io.GetInput("Voice")
	if voice == nil || voice.Input == "" {
		return ""
	}
	if voice.Tick != io.TickCounter() {
		return ""
	}
	return strings.TrimSpace(voice.Input)
}
