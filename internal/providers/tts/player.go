package tts

import (
	"context"
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

const providerQueueDepth = 8

// ttsRequest is a queued utterance with an optional per-request voice override.
type ttsRequest struct {
	text       string
	voiceID    string
	generation uint64
}

// ttsBase provides common TTS playback functionality.
type ttsBase struct {
	name         string // log prefix, e.g. "elevenlabs"
	rate         int
	outputFormat string
	log          *zap.Logger

	queue chan ttsRequest

	// synth builds and posts the provider request, streaming PCM via streamResponse.
	synth func(req ttsRequest) error
	// warmupSilenceMs, if > 0, primes the audio device with silence at startup.
	warmupSilenceMs int
	// preRollSilenceMs, if > 0, plays silence before each utterance.
	preRollSilenceMs int

	ffplayMu sync.Mutex
	ffplay   *exec.Cmd
	ffplayIn io.WriteCloser

	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

// Start launches the processAudio goroutine.
func (p *ttsBase) Start() {
	p.wg.Add(1)
	go p.processAudio()
}

// AddText enqueues text for TTS synthesis using the provider's default voice.
func (p *ttsBase) AddText(text string) {
	p.AddTextWithVoice(text, "")
}

// AddTextWithVoice enqueues text for TTS synthesis with a specific voice ID if supported by the provider.
func (p *ttsBase) AddTextWithVoice(text, voiceID string) {
	if Suppressed.Load() {
		p.log.Info(p.name+": TTS suppressed, dropping", zap.String("text", text))
		return
	}
	select {
	case p.queue <- ttsRequest{text: text, voiceID: voiceID, generation: generation.Load()}:
		pending.Add(1)
	default:
		p.log.Warn(p.name+": queue full, dropping", zap.String("text", text))
	}
}

// Stop cancels the provider context and waits for goroutines to exit.
func (p *ttsBase) Stop() {
	p.cancel()
	p.wg.Wait()
	p.cleanupFFPlay()
}

// processAudio dequeues text, synthesizes it, and streams audio to ffplay.
func (p *ttsBase) processAudio() {
	defer p.wg.Done()

	if p.warmupSilenceMs > 0 {
		p.warmUp()
	}

	for {
		select {
		case <-p.ctx.Done():
			return
		case req, ok := <-p.queue:
			if !ok {
				return
			}
			p.handleRequest(req)
		}
	}
}

// handleRequest synthesizes and plays a single dequeued utterance. It always
// clears the request's pending count when done, so pending stays positive for
// the whole lifetime of the utterance (including the gap before Speaking is
// set) and Busy reflects queued-but-unplayed speech.
func (p *ttsBase) handleRequest(req ttsRequest) {
	defer pending.Add(-1)

	if req.generation < generation.Load() {
		return
	}

	if !p.initFFPlay() {
		p.log.Error(p.name + ": ffplay unavailable, dropping utterance")
		return
	}

	if p.preRollSilenceMs > 0 {
		p.streamChunk(silenceBytes(p.rate, p.preRollSilenceMs))
	}

	Speaking.Store(true)
	defer Speaking.Store(false)

	err := p.synth(req)
	if Interrupt.Load() {
		p.handleInterrupt()
		return
	}

	if err != nil && p.ctx.Err() == nil {
		p.log.Error(p.name+": synthesis failed", zap.Error(err))
	}

	p.finishPlayback()

	if Interrupt.Load() {
		p.handleInterrupt()
	}
}

// warmUp primes the audio device by playing a short burst of silence through ffplay.
func (p *ttsBase) warmUp() {
	if p.ctx.Err() != nil {
		return
	}
	if !p.initFFPlay() {
		p.log.Warn(p.name + ": warm-up skipped, ffplay unavailable")
		return
	}
	p.streamChunk(silenceBytes(p.rate, p.warmupSilenceMs))
	p.finishPlayback()
	p.log.Debug(p.name + ": audio device warmed up")
}

// postAndStream sends req, validates the response, records HTTP timing, and
// streams the PCM body to ffplay. modelID and metricURL label the latency metrics.
func (p *ttsBase) postAndStream(req *http.Request, modelID, metricURL string) error {
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
		if Interrupt.Load() {
			return nil
		}
		n, rerr := resp.Body.Read(buf)
		if n > 0 {
			if firstChunk {
				firstChunk = false
				latency := time.Since(start).Seconds()
				metrics.TTSLatency.WithLabelValues(modelID, metricURL).Observe(latency)
				metrics.TTSLatencyLast.WithLabelValues(modelID, metricURL).Set(latency)
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

// handleInterrupt stops playback and clears the queue when an interrupt signal is received.
func (p *ttsBase) handleInterrupt() {
	p.log.Info(p.name + ": TTS interrupted by user speech")
	p.cleanupFFPlay()
	Interrupt.Store(false)
}

// initFFPlay starts a new ffplay process if one is not already running.
func (p *ttsBase) initFFPlay() bool {
	p.ffplayMu.Lock()
	defer p.ffplayMu.Unlock()

	if p.ffplay != nil && p.ffplay.ProcessState == nil {
		return true
	}

	var args []string
	if strings.Contains(p.outputFormat, "pcm") {
		args = append(args, "-f", "s16le", "-ar", strconv.Itoa(p.rate))
	}
	args = append(args, "-nodisp", "-autoexit", "-")

	cmd := exec.CommandContext(p.ctx, "ffplay", args...)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		p.log.Error(p.name+": ffplay stdin pipe", zap.Error(err))
		return false
	}
	if err := cmd.Start(); err != nil {
		p.log.Error(p.name+": ffplay start", zap.Error(err))
		return false
	}

	p.ffplay = cmd
	p.ffplayIn = stdin
	p.log.Debug(p.name + ": ffplay started")
	return true
}

// streamChunk writes a PCM chunk to the persistent ffplay stdin.
func (p *ttsBase) streamChunk(chunk []byte) {
	p.ffplayMu.Lock()
	defer p.ffplayMu.Unlock()

	if p.ffplayIn == nil {
		return
	}
	if _, err := p.ffplayIn.Write(chunk); err != nil {
		p.log.Warn(p.name+": ffplay write failed, reinitializing", zap.Error(err))
		p.cleanupFFPlayLocked()
	}
}

// finishPlayback closes ffplay stdin and waits for the process to drain and exit.
func (p *ttsBase) finishPlayback() {
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

	poll := time.NewTicker(50 * time.Millisecond)
	defer poll.Stop()
	timeout := time.After(10 * time.Second)
	for {
		select {
		case <-done:
			p.log.Debug(p.name + ": ffplay finished")
			p.ffplay = nil
			return
		case <-poll.C:
			if Interrupt.Load() {
				_ = p.ffplay.Process.Kill()
				<-done
				p.ffplay = nil
				return
			}
		case <-timeout:
			p.log.Warn(p.name + ": ffplay timeout, killing")
			_ = p.ffplay.Process.Kill()
			<-done
			p.ffplay = nil
			return
		}
	}
}

// cleanupFFPlay force-terminates ffplay from outside the lock.
func (p *ttsBase) cleanupFFPlay() {
	p.ffplayMu.Lock()
	defer p.ffplayMu.Unlock()
	p.cleanupFFPlayLocked()
}

// cleanupFFPlayLocked force-terminates ffplay. Must be called with ffplayMu held.
func (p *ttsBase) cleanupFFPlayLocked() {
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
