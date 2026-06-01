package logger

import "go.uber.org/zap"

var gLogger *zap.Logger

func init() {
	gLogger, _ = zap.NewProduction()
}

func Set(l *zap.Logger) {
	gLogger = l
}

func Get() *zap.Logger {
	return gLogger
}

func BuildLogger(level string) *zap.Logger {
	cfg := zap.NewProductionConfig()

	if err := cfg.Level.UnmarshalText([]byte(level)); err != nil {
		cfg.Level = zap.NewAtomicLevelAt(zap.InfoLevel)
	}

	logger, _ := cfg.Build()

	return logger
}
