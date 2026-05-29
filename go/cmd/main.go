package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/config"
	"github.com/openmind/om1/internal/runtime"

	_ "github.com/openmind/om1/plugins/actions"
	_ "github.com/openmind/om1/plugins/inputs"
	_ "github.com/openmind/om1/plugins/llm"
)

func main() {
	var (
		configName = flag.String("config", "", "config name or path (required)")
		logLevel   = flag.String("log-level", "info", "log level: debug|info|warn|error")
		hotReload  = flag.Bool("hot-reload", false, "reload config on file change")
		checkSecs  = flag.Float64("check-interval", 1.0, "hot-reload check interval (seconds)")
	)
	flag.Parse()

	if *configName == "" {
		fmt.Fprintln(os.Stderr, "error: --config is required")
		flag.Usage()
		os.Exit(1)
	}

	logger := buildLogger(*logLevel)
	defer func() { _ = logger.Sync() }()

	cfg, err := config.Load(*configName)
	if err != nil {
		logger.Fatal("failed to load config", zap.Error(err))
	}

	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer cancel()

	rt := runtime.New(cfg, logger, runtime.Options{
		HotReload:     *hotReload,
		CheckInterval: *checkSecs,
	})

	if err := rt.Run(ctx); err != nil && err != context.Canceled {
		logger.Fatal("runtime exited with error", zap.Error(err))
	}
}

func buildLogger(level string) *zap.Logger {
	cfg := zap.NewProductionConfig()
	if err := cfg.Level.UnmarshalText([]byte(level)); err != nil {
		cfg.Level = zap.NewAtomicLevelAt(zap.InfoLevel)
	}
	l, _ := cfg.Build()
	return l
}
