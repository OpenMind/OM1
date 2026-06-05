package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"syscall"
	"time"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/config"
	"github.com/openmind/om1/internal/logger"
	"github.com/openmind/om1/internal/metrics"
	"github.com/openmind/om1/internal/runtime"

	_ "github.com/openmind/om1/plugins/actions"
	_ "github.com/openmind/om1/plugins/backgrounds"
	_ "github.com/openmind/om1/plugins/inputs"
	_ "github.com/openmind/om1/plugins/llm"
)

func main() {
	processStart := time.Now()

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

	log := logger.BuildLogger(*logLevel)
	logger.Set(log)
	defer func() { _ = log.Sync() }()

	stopMetrics := metrics.StartServer(log)
	defer func() {
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		_ = stopMetrics(shutdownCtx)
	}()

	cfg, err := config.Load(*configName)
	if err != nil {
		log.Fatal("failed to load config", zap.Error(err))
	}

	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer cancel()

	rt := runtime.New(cfg, log, runtime.Options{
		HotReload:     *hotReload,
		CheckInterval: *checkSecs,
	})

	// Record cold/warm start time: process start -> runtime ready (about to run).
	metrics.RecordStartupComplete(processStart)

	if err := rt.Run(ctx); err != nil && err != context.Canceled {
		log.Fatal("runtime exited with error", zap.Error(err))
	}
}
