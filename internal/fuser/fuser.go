package fuser

import (
	"context"
	"strings"

	"github.com/openmind/om1/internal/actions"
	"github.com/openmind/om1/internal/config"
	"github.com/openmind/om1/internal/memory"
	"github.com/openmind/om1/internal/providers"
	"go.uber.org/zap"
)

// Fuser is responsible for fusing various prompt components (system persona, sensory inputs, knowledge base context, available actions, examples) into a single prompt string to be sent to the LLM.
type Fuser struct {
	runtimeConfig *config.RuntimeConfig
	agentActions  []*actions.AgentAction
	knowledgeBase KnowledgeBase
	memory        *memory.Manager
	log           *zap.Logger
}

// KnowledgeBase defines an interface for querying a knowledge base with a question and retrieving relevant documents.
type KnowledgeBase interface {
	Query(ctx context.Context, question string, topK int) ([]string, error)
}

// NewFuser constructs a Fuser with the given runtime configuration, agent actions, knowledge base, and logger.
func NewFuser(runtimeConfig *config.RuntimeConfig, agentActions []*actions.AgentAction, knowledgeBase KnowledgeBase, memory *memory.Manager, log *zap.Logger) *Fuser {
	return &Fuser{runtimeConfig: runtimeConfig, agentActions: agentActions, knowledgeBase: knowledgeBase, memory: memory, log: log}
}

// Fuse combines the prompt components into a single string to be sent to the LLM.
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

	// 3a. Knowledge-base context (RAG).
	if f.knowledgeBase != nil && f.runtimeConfig.KnowledgeBase != nil {
		if question := f.voiceQuery(); question != "" {
			documents, err := f.knowledgeBase.Query(ctx, question, f.runtimeConfig.KnowledgeBase.TopK)
			if err != nil {
				f.log.Warn("knowledge base query failed", zap.Error(err))
			} else if len(documents) > 0 {
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

	// 3b. Long-term memory context.
	if f.memory != nil {
		user := memory.ResolveCurrentUser()
		providers.IO().SetDynamicVar("current_user_id", user.UUID)
		providers.IO().SetDynamicVar("current_user_name", user.Name)

		if question := f.voiceQuery(); question != "" {
			memCtx := f.memory.SearchAndFormat(ctx, question, user.UUID)
			if memCtx != "" {
				builder.WriteString("MEMORY:\n")
				builder.WriteString(memCtx)
				builder.WriteString("\n\nProactively reference MEMORY in your responses. Prioritize it over your own knowledge.\n\n")
				f.log.Info("memory: injecting context", zap.Int("chars", len(memCtx)), zap.String("uuid", user.UUID))
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

// voiceQuery retrieves the latest voice input from the IOProvider for the current tick.
func (f *Fuser) voiceQuery() string {
	io := providers.IO()
	voice := io.GetInput("Voice")
	if voice == nil || voice.Input == "" || voice.Tick != io.TickCounter() {
		return ""
	}
	return strings.TrimSpace(voice.Input)
}
