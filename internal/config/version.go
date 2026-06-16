package config

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/openmind/om1/internal/logger"
	"go.uber.org/zap"
)

const LatestRuntimeVersion = "v1.1.0"

func GetRuntimeVersion() string {
	return LatestRuntimeVersion
}

func parseVersion(version string) ([3]int, error) {
	var parts [3]int

	trimmed := strings.TrimPrefix(strings.TrimSpace(version), "v")
	segments := strings.Split(trimmed, ".")
	if len(segments) == 0 || segments[0] == "" {
		return parts, fmt.Errorf("invalid version format")
	}
	if len(segments) > 3 {
		return parts, fmt.Errorf("invalid version format")
	}

	for i, segment := range segments {
		n, err := strconv.Atoi(segment)
		if err != nil {
			return parts, fmt.Errorf("invalid version format")
		}
		parts[i] = n
	}

	return parts, nil
}

func IsVersionSupported(version string) (bool, error) {
	if version == "" {
		return false, fmt.Errorf("version cannot be empty")
	}

	supported, err := parseVersion(LatestRuntimeVersion)
	if err != nil {
		return false, err
	}

	input, err := parseVersion(version)
	if err != nil {
		return false, err
	}

	if supported[0] != input[0] {
		return false, fmt.Errorf("major version mismatch: expected %d, got %d", supported[0], input[0])
	}

	if supported[1] != input[1] {
		logger.Get().Warn("version mismatch may cause compatibility issues",
			zap.Int("expected_minor", supported[1]),
			zap.Int("got_minor", input[1]),
		)
	}

	return true, nil
}

func VerifyRuntimeVersion(configVersion, configName string) (bool, error) {
	if configName == "" {
		configName = "configuration"
	}

	runtimeVersion := GetRuntimeVersion()

	log := logger.Get()
	log.Info("loading configuration",
		zap.String("config", configName),
		zap.String("config_version", configVersion),
		zap.String("runtime_version", runtimeVersion),
	)

	supported, err := IsVersionSupported(configVersion)
	if err != nil {
		log.Error("version compatibility check failed",
			zap.String("config", configName),
			zap.Error(err),
		)
		return false, fmt.Errorf(
			"configuration version %q is incompatible with runtime version %q: %w",
			configVersion, runtimeVersion, err,
		)
	}

	if supported {
		log.Info("config version is compatible with runtime", zap.String("config", configName))
	}

	return supported, nil
}
