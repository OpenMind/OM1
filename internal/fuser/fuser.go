package fuser

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/openmind/om1/internal/actions"
	"github.com/openmind/om1/internal/config"
	"github.com/openmind/om1/internal/inputs"
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
	System       string
	User         string
	Summary      string
	Observations string // this tick's new conversation and event text joined, without the time; "" on an idle tick
	// HasConversation / HasEvent say whether this tick carries a new conversation turn or a new event. World
	// state alone sets neither: it keeps the robot aware but is not a reason to respond.
	HasConversation bool
	HasEvent        bool
}

// Fuse combines the prompt components into a single string to be sent to the LLM.
func (f *Fuser) Fuse(ctx context.Context, sensorBuffers []string) (string, error) {
	p, err := f.FuseParts(ctx, sensorBuffers)
	if err != nil {
		return "", err
	}
	return p.System + "\n\n" + p.User, nil
}

// FuseParts builds the prompt as separate system and user parts (see Prompt). Plain buffers carry no role and
// are treated as events, which is what every input was before roles existed.
func (f *Fuser) FuseParts(ctx context.Context, sensorBuffers []string) (Prompt, error) {
	obs := make([]inputs.Observation, 0, len(sensorBuffers))
	for _, b := range sensorBuffers {
		if b != "" {
			obs = append(obs, inputs.Observation{Role: inputs.RoleEvent, Text: strings.TrimSpace(b), New: true})
		}
	}
	return f.FuseObservations(ctx, obs)
}

// FuseObservations builds the prompt from role-tagged observations. Conversation turns and events are listed as
// what happened since the last action; world state is a standing section the model is told not to narrate.
func (f *Fuser) FuseObservations(ctx context.Context, observations []inputs.Observation) (Prompt, error) {
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
	sys.WriteString("Each turn lists what is new since your last action: what someone said to you, and events. " +
		"World state is shown separately for your awareness; it is not a reason to speak or act by itself. " +
		"Observations you already responded to appear in the conversation history together with what you did; " +
		"do not respond to them again.")

	// User part: time, this tick's observations, retrieved context, closing question.
	now := time.Now()
	usr.WriteString("Current time is ")
	usr.WriteString(now.Format("Monday, January 2, 2006 at 3:04 PM"))
	usr.WriteString(".\n\n")

	var conversation, events, observed []string
	var world []inputs.Observation
	for _, o := range observations {
		if o.Text == "" {
			continue
		}
		switch o.Role {
		case inputs.RoleConversation:
			conversation = append(conversation, o.Text)
			observed = append(observed, o.Text)
		case inputs.RoleWorld:
			world = append(world, o)
		default:
			events = append(events, o.Text)
			observed = append(observed, o.Text)
		}
	}
	if len(conversation) > 0 {
		usr.WriteString("Someone said to you:\n")
		for _, c := range conversation {
			usr.WriteString("- ")
			usr.WriteString(c)
			usr.WriteString("\n")
		}
		usr.WriteString("\n")
	}
	if len(events) > 0 {
		usr.WriteString("Current observations:\n")
		for _, e := range events {
			usr.WriteString("- ")
			usr.WriteString(e)
			usr.WriteString("\n")
		}
		usr.WriteString("\n")
	}
	if len(conversation) == 0 && len(events) == 0 {
		usr.WriteString("No new observations since your last action.\n\n")
	}
	if len(world) > 0 {
		usr.WriteString("World state, for your awareness (do not narrate or react to it by itself):\n")
		for _, w := range world {
			usr.WriteString("- ")
			usr.WriteString(w.Text)
			if w.New {
				usr.WriteString(" (just updated)")
			} else {
				usr.WriteString(" (" + ageString(w.Age) + " ago)")
			}
			usr.WriteString("\n")
		}
		usr.WriteString("\n")
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

	return Prompt{System: sys.String(), User: usr.String(), Summary: summary, Observations: strings.Join(observed, "; "),
		HasConversation: len(conversation) > 0, HasEvent: len(events) > 0}, nil
}

// ageString renders a duration coarsely for the prompt ("3 s", "2 min").
func ageString(d time.Duration) string {
	if d < time.Minute {
		return fmt.Sprintf("%d s", int(d.Seconds()))
	}
	return fmt.Sprintf("%d min", int(d.Minutes()))
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
