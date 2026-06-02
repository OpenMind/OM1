package providers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/httpclient"
	"github.com/openmind/om1/internal/metrics"
)

const (
	DefaultKokoroBaseURL      = "http://127.0.0.1:8880/v1"
	DefaultKokoroVoiceID      = "af_bella"
	DefaultKokoroModelID      = "kokoro"
	DefaultKokoroOutputFormat = "pcm"
	DefaultKokoroRate         = 24000

	kokoroSpeechPath = "/audio/speech"
)

// KokoroConfig holds connection and audio parameters for the Kokoro provider.
type KokoroConfig struct {
	BaseURL      string
	APIKey       string
	VoiceID      string
	ModelID      string
	OutputFormat string
	Rate         int
}

var (
	kokoroOnce     sync.Once
	kokoroInstance *KokoroProvider
)

// Kokoro returns the singleton KokoroProvider instance, initializing it on first call.
func Kokoro(cfg KokoroConfig, log *zap.Logger) *KokoroProvider {
	kokoroOnce.Do(func() {
		kokoroInstance = newKokoroProvider(cfg, log)
		kokoroInstance.Start()
	})
	return kokoroInstance
}

// KokoroProvider manages a persistent ffplay process to stream TTS audio from a
// Kokoro (OpenAI-compatible) TTS endpoint.
type KokoroProvider struct {
	cfg       KokoroConfig
	speechURL string
	log       *zap.Logger

	queue chan string

	ffplayMu sync.Mutex
	ffplay   *exec.Cmd
	ffplayIn io.WriteCloser

	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

func newKokoroProvider(cfg KokoroConfig, log *zap.Logger) *KokoroProvider {
	ctx, cancel := context.WithCancel(context.Background())
	return &KokoroProvider{
		cfg:       cfg,
		speechURL: strings.TrimRight(cfg.BaseURL, "/") + kokoroSpeechPath,
		log:       log,
		queue:     make(chan string, providerQueueDepth),
		ctx:       ctx,
		cancel:    cancel,
	}
}

// Start launches the processAudio goroutine.
func (p *KokoroProvider) Start() {
	p.wg.Add(1)
	go p.processAudio()
}

// AddText enqueues text for TTS synthesis. Non-blocking; drops if the queue is full.
func (p *KokoroProvider) AddText(text string) {
	select {
	case p.queue <- text:
	default:
		p.log.Warn("kokoro: queue full, dropping", zap.String("text", text))
	}
}

// Stop cancels the provider context and waits for goroutines to exit.
func (p *KokoroProvider) Stop() {
	p.cancel()
	p.wg.Wait()
	p.cleanupFFPlay()
}

// processAudio dequeues text, synthesizes it, and streams audio to ffplay.
func (p *KokoroProvider) processAudio() {
	defer p.wg.Done()
	for {
		select {
		case <-p.ctx.Done():
			return
		case text, ok := <-p.queue:
			if !ok {
				return
			}
			if !p.initFFPlay() {
				p.log.Error("kokoro: ffplay unavailable, dropping utterance")
				continue
			}
			p.streamChunk(silenceBytes(p.cfg.Rate, 10))

			Speaking.Store(true)
			if err := p.synthesize(text); err != nil && p.ctx.Err() == nil {
				p.log.Error("kokoro: synthesis failed", zap.Error(err))
			}
			p.finishPlayback()
			Speaking.Store(false)
		}
	}
}

// synthesize posts text to the Kokoro endpoint and streams PCM chunks to ffplay.
func (p *KokoroProvider) synthesize(text string) error {
	body := map[string]any{
		"model":           p.cfg.ModelID,
		"voice":           p.cfg.VoiceID,
		"response_format": p.cfg.OutputFormat,
		"input":           text,
	}

	bodyBytes, err := json.Marshal(body)
	if err != nil {
		return fmt.Errorf("marshal: %w", err)
	}

	req, err := http.NewRequestWithContext(p.ctx, http.MethodPost, p.speechURL, bytes.NewReader(bodyBytes))
	if err != nil {
		return fmt.Errorf("request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if p.cfg.APIKey != "" {
		req.Header.Set("x-api-key", p.cfg.APIKey)
	}

	start := time.Now()
	resp, err := httpclient.Default().Do(req)
	if err != nil {
		return fmt.Errorf("http: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("api %d: %s", resp.StatusCode, b)
	}

	metrics.RecordHTTPTiming(req.URL.Host, req.URL.Path, req.Method, resp.StatusCode,
		resp.Header.Get("X-Proxy-Parse-Ms"), resp.Header.Get("X-Upstream-Total-Ms"),
		resp.Header.Get("X-Upstream-TTFB-Ms"), resp.Header.Get("X-Proxy-Total-Ms"))

	buf := make([]byte, 1024)
	firstChunk := true
	for {
		n, rerr := resp.Body.Read(buf)
		if n > 0 {
			if firstChunk {
				firstChunk = false
				latency := time.Since(start).Seconds()
				metrics.TTSLatency.WithLabelValues(p.cfg.ModelID, p.speechURL).Observe(latency)
				metrics.TTSLatencyLast.WithLabelValues(p.cfg.ModelID, p.speechURL).Set(latency)
			}
			p.streamChunk(buf[:n])
		}
		if rerr == io.EOF {
			break
		}
		if rerr != nil {
			return fmt.Errorf("stream: %w", rerr)
		}
	}
	return nil
}

// initFFPlay starts a new ffplay process if one is not already running.
func (p *KokoroProvider) initFFPlay() bool {
	p.ffplayMu.Lock()
	defer p.ffplayMu.Unlock()

	if p.ffplay != nil && p.ffplay.ProcessState == nil {
		return true
	}

	var args []string
	if strings.Contains(p.cfg.OutputFormat, "pcm") {
		args = append(args, "-f", "s16le", "-ar", strconv.Itoa(p.cfg.Rate))
	}
	args = append(args, "-nodisp", "-autoexit", "-")

	cmd := exec.CommandContext(p.ctx, "ffplay", args...)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		p.log.Error("kokoro: ffplay stdin pipe", zap.Error(err))
		return false
	}
	if err := cmd.Start(); err != nil {
		p.log.Error("kokoro: ffplay start", zap.Error(err))
		return false
	}

	p.ffplay = cmd
	p.ffplayIn = stdin
	p.log.Debug("kokoro: ffplay started")
	return true
}

// streamChunk writes a PCM chunk to the persistent ffplay stdin.
func (p *KokoroProvider) streamChunk(chunk []byte) {
	p.ffplayMu.Lock()
	defer p.ffplayMu.Unlock()

	if p.ffplayIn == nil {
		return
	}
	if _, err := p.ffplayIn.Write(chunk); err != nil {
		p.log.Warn("kokoro: ffplay write failed, reinitializing", zap.Error(err))
		p.cleanupFFPlayLocked()
	}
}

// finishPlayback closes ffplay stdin and waits for the process to drain and exit.
func (p *KokoroProvider) finishPlayback() {
	p.ffplayMu.Lock()
	defer p.ffplayMu.Unlock()

	if p.ffplay == nil {
		return
	}
	if p.ffplayIn != nil {
		_ = p.ffplayIn.Close()
		p.ffplayIn = nil
	}

	done := make(chan struct{})
	go func() {
		_ = p.ffplay.Wait()
		close(done)
	}()
	select {
	case <-done:
		p.log.Debug("kokoro: ffplay finished")
	case <-time.After(10 * time.Second):
		p.log.Warn("kokoro: ffplay timeout, killing")
		_ = p.ffplay.Process.Kill()
		<-done
	}
	p.ffplay = nil
}

// cleanupFFPlay force-terminates ffplay from outside the lock.
func (p *KokoroProvider) cleanupFFPlay() {
	p.ffplayMu.Lock()
	defer p.ffplayMu.Unlock()
	p.cleanupFFPlayLocked()
}

// cleanupFFPlayLocked force-terminates ffplay. Must be called with ffplayMu held.
func (p *KokoroProvider) cleanupFFPlayLocked() {
	if p.ffplayIn != nil {
		_ = p.ffplayIn.Close()
		p.ffplayIn = nil
	}
	if p.ffplay != nil && p.ffplay.Process != nil {
		_ = p.ffplay.Process.Kill()
		_ = p.ffplay.Wait()
		p.ffplay = nil
	}
}
