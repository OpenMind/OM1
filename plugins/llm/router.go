package llm

import (
	"context"
	"fmt"
	"regexp"
	"strings"

	"github.com/openmind/om1/internal/llm"
	"github.com/openmind/om1/internal/logger"
	"go.uber.org/zap"
)

func init() {
	llm.Register("RouterLLM", NewRouter)
}

type routeConfig struct {
	Name      string         `json:"name"`
	LLMType   string         `json:"llm_type"`
	LLMConfig map[string]any `json:"llm_config"`
	Keywords  []string       `json:"keywords"`
	Patterns  []string       `json:"patterns"`
}

type routerConfig struct {
	Routes       []routeConfig `json:"routes"`
	DefaultRoute string        `json:"default_route"`
	APIKey       string        `json:"api_key"`
}

type route struct {
	name     string
	llm      llm.LLM
	keywords []string
	regexes  []*regexp.Regexp
}

func (r *route) score(text string) int {
	score := 0
	for _, kw := range r.keywords {
		if kw != "" && strings.Contains(text, kw) {
			score++
		}
	}
	for _, re := range r.regexes {
		if re.MatchString(text) {
			score++
		}
	}
	return score
}

type routerLLM struct {
	routes []*route
	def    *route
	log    *zap.Logger
}

func NewRouter(configMap map[string]any) (llm.LLM, error) {
	var cfg routerConfig
	if err := remarshal(configMap, &cfg); err != nil {
		return nil, fmt.Errorf("RouterLLM config: %w", err)
	}
	if len(cfg.Routes) == 0 {
		return nil, fmt.Errorf("RouterLLM: at least one route is required")
	}

	router := &routerLLM{log: logger.Get().Named("RouterLLM")}

	for _, rc := range cfg.Routes {
		if rc.Name == "" {
			return nil, fmt.Errorf("RouterLLM: every route needs a name")
		}
		if rc.LLMType == "" {
			return nil, fmt.Errorf("RouterLLM: route %q has no llm_type", rc.Name)
		}

		subCfg := cloneStringAnyMap(rc.LLMConfig)
		if cfg.APIKey != "" {
			if _, ok := subCfg["api_key"]; !ok {
				subCfg["api_key"] = cfg.APIKey
			}
		}

		sub, err := llm.Load(rc.LLMType, subCfg)
		if err != nil {
			return nil, fmt.Errorf("RouterLLM: load route %q (%s): %w", rc.Name, rc.LLMType, err)
		}

		regexes := make([]*regexp.Regexp, 0, len(rc.Patterns))
		for _, p := range rc.Patterns {
			re, err := regexp.Compile("(?i)" + p)
			if err != nil {
				return nil, fmt.Errorf("RouterLLM: route %q pattern %q: %w", rc.Name, p, err)
			}
			regexes = append(regexes, re)
		}

		keywords := make([]string, 0, len(rc.Keywords))
		for _, kw := range rc.Keywords {
			keywords = append(keywords, strings.ToLower(kw))
		}

		router.routes = append(router.routes, &route{
			name:     rc.Name,
			llm:      sub,
			keywords: keywords,
			regexes:  regexes,
		})
	}

	router.def = router.routes[0]
	if cfg.DefaultRoute != "" {
		found := false
		for _, r := range router.routes {
			if r.name == cfg.DefaultRoute {
				router.def = r
				found = true
				break
			}
		}
		if !found {
			return nil, fmt.Errorf("RouterLLM: default_route %q is not a defined route", cfg.DefaultRoute)
		}
	}

	return router, nil
}

func (r *routerLLM) SetSchemas(schemas []map[string]any) {
	for _, rt := range r.routes {
		rt.llm.SetSchemas(schemas)
	}
}

func (r *routerLLM) FunctionSchemas() []map[string]any { return r.def.llm.FunctionSchemas() }

func (r *routerLLM) logger() *zap.Logger {
	if r.log != nil {
		return r.log
	}
	return logger.Get().Named("RouterLLM")
}

func (r *routerLLM) pick(prompt string) *route {
	text := strings.ToLower(extractVoiceInput(prompt))
	if text == "" {
		return r.def
	}

	best := r.def
	bestScore := 0
	for _, rt := range r.routes {
		if s := rt.score(text); s > bestScore {
			best, bestScore = rt, s
		}
	}
	return best
}

func (r *routerLLM) Call(ctx context.Context, prompt string, history []llm.Message) (*llm.Response, error) {
	chosen := r.pick(prompt)
	r.logger().Info("routed", zap.String("route", chosen.name))
	return chosen.llm.Call(ctx, prompt, history)
}
