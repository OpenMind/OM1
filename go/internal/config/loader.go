package config

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

var envVarRe = regexp.MustCompile(`\$\{([^}:-]+)(?::-(.*?))?\}`)

func Load(nameOrPath string) (*SystemConfig, error) {
	path, err := resolvePath(nameOrPath)
	if err != nil {
		return nil, err
	}

	rawBytes, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read config %s: %w", path, err)
	}

	expanded := expandEnv(string(rawBytes))

	jsonData, err := stripJSON5([]byte(expanded))
	if err != nil {
		return nil, fmt.Errorf("parse config %s: %w", path, err)
	}

	var systemConfig SystemConfig
	if err := json.Unmarshal(jsonData, &systemConfig); err != nil {
		return nil, fmt.Errorf("unmarshal config %s: %w", path, err)
	}

	normalize(&systemConfig)
	return &systemConfig, nil
}

func resolvePath(nameOrPath string) (string, error) {
	if filepath.IsAbs(nameOrPath) {
		return nameOrPath, nil
	}

	if strings.HasSuffix(nameOrPath, ".json5") || strings.HasSuffix(nameOrPath, ".json") {
		return nameOrPath, nil
	}

	exe, _ := os.Executable()
	repoRoot := filepath.Join(filepath.Dir(exe), "..", "..")
	candidates := []string{
		filepath.Join(repoRoot, "config", nameOrPath+".json5"),
		filepath.Join("config", nameOrPath+".json5"),
	}

	for _, candidate := range candidates {
		if _, err := os.Stat(candidate); err == nil {
			return candidate, nil
		}
	}

	return "", fmt.Errorf("config %q not found", nameOrPath)
}

func expandEnv(text string) string {
	return envVarRe.ReplaceAllStringFunc(text, func(match string) string {
		submatch := envVarRe.FindStringSubmatch(match)
		varName, defaultValue := submatch[1], submatch[2]

		if value := os.Getenv(varName); value != "" {
			return value
		}

		return defaultValue
	})
}

func stripJSON5(src []byte) ([]byte, error) {
	lines := strings.Split(string(src), "\n")
	var stripped []string
	for _, line := range lines {
		if commentStart := commentIndex(line); commentStart >= 0 {
			line = line[:commentStart]
		}
		stripped = append(stripped, line)
	}
	text := strings.Join(stripped, "\n")

	trailingComma := regexp.MustCompile(`,(\s*[}\]])`)
	text = trailingComma.ReplaceAllString(text, "$1")

	return []byte(text), nil
}

func commentIndex(line string) int {
	inString := false
	for i := 0; i < len(line)-1; i++ {
		if line[i] == '"' {
			inString = !inString
		}
		if !inString && line[i] == '/' && line[i+1] == '/' {
			return i
		}
	}
	return -1
}

func normalize(systemConfig *SystemConfig) {
	if len(systemConfig.Modes) > 0 {
		return
	}

	modeName := systemConfig.DefaultMode
	if modeName == "" {
		modeName = systemConfig.Name
	}
	if modeName == "" {
		modeName = "default"
	}

	hertz := systemConfig.Hertz
	if hertz == 0 {
		hertz = 1.0
	}

	systemConfig.Modes = map[string]ModeConfig{
		modeName: {
			Name:             modeName,
			DisplayName:      modeName,
			Hertz:            hertz,
			SystemPromptBase: systemConfig.SystemPromptBase,
			AgentInputs:      systemConfig.AgentInputs,
			CortexLLM:        systemConfig.CortexLLM,
			AgentActions:     systemConfig.AgentActions,
			Backgrounds:      systemConfig.Backgrounds,
			MCPServers:       systemConfig.MCPServers,
			LifecycleHooks:   systemConfig.LifecycleHooks,
		},
	}
	systemConfig.DefaultMode = modeName
}
