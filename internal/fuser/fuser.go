package fuser

import (
	"context"
	"strings"
	"time"

	"github.com/openmind/om1/internal/actions"
	"github.com/openmind/om1/internal/config"
	"github.com/openmind/om1/internal/memory"
	"github.com/openmind/om1/internal/providers"
	"go.uber.org/zap"
)

// Fuser is responsible for fusing various prompt components (system persona, sensory inputs, knowledge base context, available actions, MCP tools, examples) into a single prompt string to be sent to the LLM.
type Fuser struct {
	runtimeConfig *config.RuntimeConfig
	agentActions  []*actions.AgentAction
	knowledgeBase KnowledgeBase
	memory        memory.MemoryManager
	mcp           MCPDescriber
	log           *zap.Logger
}

// KnowledgeBase defines an interface for querying a knowledge base with a question and retrieving relevant documents.
type KnowledgeBase interface {
	Query(ctx context.Context, question string, topK int) ([]string, error)
}

// MCPDescriber supplies a prompt block describing the available MCP tools.
type MCPDescriber interface {
	ToolDescriptions() string
}

// NewFuser creates a new Fuser with the given runtime configuration, agent actions, knowledge base, memory manager, MCP describer, and logger.
func NewFuser(runtimeConfig *config.RuntimeConfig, agentActions []*actions.AgentAction, knowledgeBase KnowledgeBase, memory memory.MemoryManager, mcp MCPDescriber, log *zap.Logger) *Fuser {
	return &Fuser{runtimeConfig: runtimeConfig, agentActions: agentActions, knowledgeBase: knowledgeBase, memory: memory, mcp: mcp, log: log}
}

// Prompt is a fused prompt split into the part that is constant across ticks (System) and the part that
// changes every tick (User), plus a one-line Summary of the tick's observations for the conversation history.
//
// Keeping the two apart lets the LLM orchestrator send the constant part once as the system message and store
// only the Summary per turn, instead of a full copy of the prompt per turn.
type Prompt struct {
	System  string
	User    string
	Summary string
}

// Fuse combines the prompt components into a single string to be sent to the LLM.
func (f *Fuser) Fuse(ctx context.Context, sensorBuffers []string) (string, error) {
	p, err := f.FuseParts(ctx, sensorBuffers)
	if err != nil {
		return "", err
	}
	return p.System + "\n\n" + p.User, nil
}

// FuseParts builds the prompt as separate system and user parts (see Prompt).
func (f *Fuser) FuseParts(ctx context.Context, sensorBuffers []string) (Prompt, error) {
	var sys, usr strings.Builder

	// System part: persona, governance, available actions, MCP tools, examples. Constant across ticks.
	sys.WriteString(f.runtimeConfig.SystemPromptBase)
	sys.WriteString("\n\n")
	if f.runtimeConfig.SystemGovernance != "" {
		sys.WriteString("Governance rules:\n")
		sys.WriteString(f.runtimeConfig.SystemGovernance)
		sys.WriteString("\n\n")
	}
	visibleActions := f.visibleActions()
	if len(visibleActions) > 0 {
		sys.WriteString("Available actions:\n")
		for _, action := range visibleActions {
			sys.WriteString("- ")
			sys.WriteString(action.LLMLabel)
			sys.WriteString("\n")
		}
		sys.WriteString("\n")
	}
	if f.mcp != nil {
		if descriptions := f.mcp.ToolDescriptions(); descriptions != "" {
			sys.WriteString(descriptions)
			sys.WriteString("\n\n")
		}
	}
	if f.runtimeConfig.PromptExamples != "" {
		sys.WriteString(f.runtimeConfig.PromptExamples)
		sys.WriteString("\n\n")
	}
	sys.WriteString("Each turn lists the observations that are new since your last action. Observations you already " +
		"responded to appear in the conversation history together with what you did; do not respond to them again.")

	// User part: time, this tick's observations, retrieved context, closing question.
	now := time.Now()
	usr.WriteString("Current time is ")
	usr.WriteString(now.Format("Monday, January 2, 2006 at 3:04 PM"))
	usr.WriteString(".\n\n")

	var observed []string
	for _, buffer := range sensorBuffers {
		if buffer != "" {
			observed = append(observed, strings.TrimSpace(buffer))
		}
	}
	if len(observed) > 0 {
		usr.WriteString("Current observations:\n")
		for _, o := range observed {
			usr.WriteString("- ")
			usr.WriteString(o)
			usr.WriteString("\n")
		}
		usr.WriteString("\n")
	} else {
		usr.WriteString("No new observations since your last action.\n\n")
	}

	if f.knowledgeBase != nil && f.runtimeConfig.KnowledgeBase != nil {
		if question := f.voiceQuery(); question != "" {
			documents, err := f.knowledgeBase.Query(ctx, question, f.runtimeConfig.KnowledgeBase.TopK)
			if err != nil {
				f.log.Warn("knowledge base query failed", zap.Error(err))
			} else if len(documents) > 0 {
				usr.WriteString("Relevant context:\n")
				for _, document := range documents {
					usr.WriteString("- ")
					usr.WriteString(document)
					usr.WriteString("\n")
				}
				usr.WriteString("\n")
			}
		}
	}

	if f.memory != nil {
		user := memory.ResolveCurrentUser()
		if question := f.voiceQuery(); question != "" {
			memContext := f.memory.SearchAndFormat(ctx, question, user.UUID)
			if memContext != "" {
				usr.WriteString("MEMORY:\n")
				usr.WriteString(memContext)
				usr.WriteString("\n\nProactively reference MEMORY in your responses. Prioritize it over your own knowledge.\n\n")
				f.log.Info("memory: injecting context", zap.Int("chars", len(memContext)), zap.String("uuid", user.UUID))
			}
		}
	}

	usr.WriteString("What will you do next?")

	summary := "At " + now.Format("15:04:05") + " there were no new observations."
	if len(observed) > 0 {
		summary = "At " + now.Format("15:04:05") + " you observed: " + strings.Join(observed, "; ")
	}

	return Prompt{System: sys.String(), User: usr.String(), Summary: summary}, nil
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
