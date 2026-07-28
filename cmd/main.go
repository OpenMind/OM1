package main

import (
	"cmp"
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
	"github.com/openmind/om1/internal/providers"
	"github.com/openmind/om1/internal/qualityscorer"
	"github.com/openmind/om1/internal/runtime"

	_ "github.com/openmind/om1/plugins/actions"
	_ "github.com/openmind/om1/plugins/backgrounds"
	_ "github.com/openmind/om1/plugins/inputs"
	_ "github.com/openmind/om1/plugins/llm"
)

func main() {
	var (
		configName = flag.String("config", "", "config name or path (required)")
		logLevel   = flag.String("log-level", cmp.Or(os.Getenv("LOG_LEVEL"), "info"), "log level: debug|info|warn|error (env: LOG_LEVEL)")
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

	if cfg.UseTracer != nil && cfg.UseTracer.QualityScorer != nil && cfg.UseTracer.QualityScorer.Enabled {
		if !cfg.UseTracer.Enabled {
			log.Warn("use_tracer.quality_scorer.enabled is true but use_tracer.enabled is false -- it will never receive any trace records")
		}
		qsCfg := *cfg.UseTracer.QualityScorer
		if qsCfg.APIKey == "" {
			// Same fallback every LLM plugin gets via buildSystemMeta/addMeta (internal/runtime/config.go).
			qsCfg.APIKey = cfg.APIKey
		}
		stopScorer := qualityscorer.Start(ctx, log, providers.TracerProvider(), qsCfg)
		defer stopScorer()
	}

	rt := runtime.New(cfg, log, runtime.Options{
		HotReload:     *hotReload,
		CheckInterval: *checkSecs,
	})

	if err := rt.Run(ctx); err != nil && err != context.Canceled {
		log.Fatal("runtime exited with error", zap.Error(err))
	}
}
