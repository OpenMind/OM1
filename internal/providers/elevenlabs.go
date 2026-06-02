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
	ElevenLabsTTSURL    = "https://api.openmind.com/api/core/elevenlabs/tts/audio/speech"
	DefaultVoiceID      = "JBFqnCBsd6RMkjVDRZzb"
	DefaultModelID      = "eleven_flash_v2_5"
	DefaultOutputFormat = "pcm_16000"
	DefaultRate         = 16000

	providerQueueDepth = 8
)

// ElevenLabsConfig holds connection and audio parameters for the provider.
type ElevenLabsConfig struct {
	APIKey           string
	ElevenLabsAPIKey string
	VoiceID          string
	ModelID          string
	OutputFormat     string
	Rate             int
}

var (
	elevenLabsOnce     sync.Once
	elevenLabsInstance *ElevenLabsProvider
)

// ttsRequest is a queued utterance with an optional per-request voice override.
type ttsRequest struct {
	text    string
	voiceID string
}

// ElevenLabs returns the singleton ElevenLabsProvider instance, initializing it on first call.
func ElevenLabs(cfg ElevenLabsConfig, log *zap.Logger) *ElevenLabsProvider {
	elevenLabsOnce.Do(func() {
		elevenLabsInstance = newElevenLabsProvider(cfg, log)
		elevenLabsInstance.Start()
	})
	return elevenLabsInstance
}

// ElevenLabsProvider manages a persistent ffplay process to stream TTS audio from ElevenLabs.
type ElevenLabsProvider struct {
	cfg ElevenLabsConfig
	log *zap.Logger

	queue chan ttsRequest

	ffplayMu sync.Mutex
	ffplay   *exec.Cmd
	ffplayIn io.WriteCloser

	lastAudioMu   sync.Mutex
	lastAudioTime time.Time

	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

func newElevenLabsProvider(cfg ElevenLabsConfig, log *zap.Logger) *ElevenLabsProvider {
	ctx, cancel := context.WithCancel(context.Background())
	return &ElevenLabsProvider{
		cfg:           cfg,
		log:           log,
		queue:         make(chan ttsRequest, providerQueueDepth),
		lastAudioTime: time.Now(),
		ctx:           ctx,
		cancel:        cancel,
	}
}

// Start launches the processAudio.
func (p *ElevenLabsProvider) Start() {
	p.wg.Add(1)
	go p.processAudio()
}

// AddText enqueues text for TTS synthesis using the provider's default voice.
// Non-blocking; drops if the queue is full.
func (p *ElevenLabsProvider) AddText(text string) {
	p.AddTextWithVoice(text, "")
}

// AddTextWithVoice enqueues text for TTS synthesis with an optional per-utterance
// voice override. An empty voiceID falls back to the provider's default VoiceID.
// Non-blocking; drops if the queue is full.
func (p *ElevenLabsProvider) AddTextWithVoice(text, voiceID string) {
	select {
	case p.queue <- ttsRequest{text: text, voiceID: voiceID}:
	default:
		p.log.Warn("elevenlabs: queue full, dropping", zap.String("text", text))
	}
}

// Stop cancels the provider context and waits for goroutines to exit.
func (p *ElevenLabsProvider) Stop() {
	p.cancel()
	p.wg.Wait()
	p.cleanupFFPlay()
}

// processAudio dequeues text, synthesizes it, and streams audio to ffplay.
func (p *ElevenLabsProvider) processAudio() {
	defer p.wg.Done()
	for {
		select {
		case <-p.ctx.Done():
			return
		case req, ok := <-p.queue:
			if !ok {
				return
			}
			if !p.initFFPlay() {
				p.log.Error("elevenlabs: ffplay unavailable, dropping utterance")
				continue
			}
			p.streamChunk(silenceBytes(p.cfg.Rate, 10))

			Speaking.Store(true)
			if err := p.synthesize(req.text, req.voiceID); err != nil && p.ctx.Err() == nil {
				p.log.Error("elevenlabs: synthesis failed", zap.Error(err))
			}
			p.finishPlayback()
			Speaking.Store(false)
		}
	}
}

// synthesize posts text to the ElevenLabs endpoint and streams PCM chunks to ffplay.
// An empty voiceID falls back to the provider's configured default voice.
func (p *ElevenLabsProvider) synthesize(text, voiceID string) error {
	if voiceID == "" {
		voiceID = p.cfg.VoiceID
	}
	body := map[string]any{
		"model":           p.cfg.ModelID,
		"voice":           voiceID,
		"response_format": p.cfg.OutputFormat,
		"input":           text,
	}
	if p.cfg.ElevenLabsAPIKey != "" {
		body["elevenlabs_api_key"] = p.cfg.ElevenLabsAPIKey
	}

	bodyBytes, err := json.Marshal(body)
	if err != nil {
		return fmt.Errorf("marshal: %w", err)
	}

	req, err := http.NewRequestWithContext(p.ctx, http.MethodPost, ElevenLabsTTSURL, bytes.NewReader(bodyBytes))
	if err != nil {
		return fmt.Errorf("request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", p.cfg.APIKey)

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
				metrics.TTSLatency.WithLabelValues(p.cfg.ModelID, ElevenLabsTTSURL).Observe(latency)
				metrics.TTSLatencyLast.WithLabelValues(p.cfg.ModelID, ElevenLabsTTSURL).Set(latency)
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
func (p *ElevenLabsProvider) initFFPlay() bool {
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
		p.log.Error("elevenlabs: ffplay stdin pipe", zap.Error(err))
		return false
	}
	if err := cmd.Start(); err != nil {
		p.log.Error("elevenlabs: ffplay start", zap.Error(err))
		return false
	}

	p.ffplay = cmd
	p.ffplayIn = stdin
	p.log.Debug("elevenlabs: ffplay started")
	return true
}

// streamChunk writes a PCM chunk to the persistent ffplay stdin.
func (p *ElevenLabsProvider) streamChunk(chunk []byte) {
	p.ffplayMu.Lock()
	defer p.ffplayMu.Unlock()

	if p.ffplayIn == nil {
		return
	}
	if _, err := p.ffplayIn.Write(chunk); err != nil {
		p.log.Warn("elevenlabs: ffplay write failed, reinitializing", zap.Error(err))
		p.cleanupFFPlayLocked()
		return
	}
	p.lastAudioMu.Lock()
	p.lastAudioTime = time.Now()
	p.lastAudioMu.Unlock()
}

// finishPlayback closes ffplay stdin and waits for the process to drain and exit.
func (p *ElevenLabsProvider) finishPlayback() {
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
		p.log.Debug("elevenlabs: ffplay finished")
	case <-time.After(10 * time.Second):
		p.log.Warn("elevenlabs: ffplay timeout, killing")
		_ = p.ffplay.Process.Kill()
		<-done
	}
	p.ffplay = nil
}

// cleanupFFPlay force-terminates ffplay from outside the lock.
func (p *ElevenLabsProvider) cleanupFFPlay() {
	p.ffplayMu.Lock()
	defer p.ffplayMu.Unlock()
	p.cleanupFFPlayLocked()
}

// cleanupFFPlayLocked force-terminates ffplay. Must be called with ffplayMu held.
func (p *ElevenLabsProvider) cleanupFFPlayLocked() {
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

// silenceBytes generates silent 16-bit PCM audio for the given duration.
func silenceBytes(rate, ms int) []byte {
	return make([]byte, rate*ms/1000*2)
}
