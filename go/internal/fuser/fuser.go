package fuser

import (
	"context"
	"strings"

	"github.com/openmind/om1/internal/actions"
	"github.com/openmind/om1/internal/config"
)

type Fuser struct {
	runtimeConfig *config.RuntimeConfig
	agentActions  []*actions.AgentAction
	knowledgeBase KnowledgeBase
}

type KnowledgeBase interface {
	Query(ctx context.Context, question string, topK int) ([]string, error)
}

func New(runtimeConfig *config.RuntimeConfig, agentActions []*actions.AgentAction, knowledgeBase KnowledgeBase) *Fuser {
	return &Fuser{runtimeConfig: runtimeConfig, agentActions: agentActions, knowledgeBase: knowledgeBase}
}

func (f *Fuser) Fuse(ctx context.Context, sensorBuffers []string) (string, error) {
	var builder strings.Builder

	// 1. System persona + governance.
	builder.WriteString(f.runtimeConfig.SystemPromptBase)
	builder.WriteString("\n\n")
	if f.runtimeConfig.SystemGovernance != "" {
		builder.WriteString("Governance rules:\n")
		builder.WriteString(f.runtimeConfig.SystemGovernance)
		builder.WriteString("\n\n")
	}

	// 2. Sensory inputs.
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

	// 3. Knowledge-base context (RAG).
	if f.knowledgeBase != nil && f.runtimeConfig.KnowledgeBase != nil {
		question := f.buildQuestion()
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

func (f *Fuser) visibleActions() []*actions.AgentAction {
	var visible []*actions.AgentAction
	for _, action := range f.agentActions {
		if !action.ExcludeFromPrompt {
			visible = append(visible, action)
		}
	}
	return visible
}

func (f *Fuser) buildQuestion() string {
	return "What should " + f.runtimeConfig.Name + " do next?"
}
