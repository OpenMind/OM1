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
