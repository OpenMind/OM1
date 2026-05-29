package speak

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

	"github.com/openmind/om1/internal/actions"
	"github.com/openmind/om1/internal/httpclient"
	"github.com/openmind/om1/internal/providers"
)

type SpeakInput struct {
	Action string `json:"action" description:"The text to be spoken"`
}

func init() {
	actions.RegisterInterface(
		"speak",
		"Action interface for text-to-speech output. Makes the robot speak the provided text.",
		SpeakInput{},
	)
	actions.Register("speak/elevenlabs_tts", NewElevenLabsTTS)
}

const (
	defaultTTSBaseURL   = "https://api.openmind.com/api/core/elevenlabs/tts"
	defaultVoiceID      = "JBFqnCBsd6RMkjVDRZzb"
	defaultModelID      = "eleven_flash_v2_5"
	defaultOutputFormat = "pcm_16000"
	defaultRate         = 16000
	ttsQueueDepth       = 8
	keepaliveInterval   = 60 * time.Second
)

type ElevenLabsConfig struct {
	APIKey           string `json:"api_key"`
	ElevenLabsAPIKey string `json:"elevenlabs_api_key"`
	VoiceID          string `json:"voice_id"`
	ModelID          string `json:"model_id"`
	OutputFormat     string `json:"output_format"`
	Rate             int    `json:"rate"`
	SilenceRate      int    `json:"silence_rate"`
}

type TTSRequest struct {
	text    string
	voiceID string
}

type ElevenLabsConnector struct {
	cfg    ElevenLabsConfig
	log    *zap.Logger
	ctx    context.Context
	cancel context.CancelFunc

	queue chan TTSRequest
	wg    sync.WaitGroup

	ffplayMu sync.Mutex
	ffplay   *exec.Cmd
	ffplayIn io.WriteCloser

	silenceAudio  []byte
	lastAudioMu   sync.Mutex
	lastAudioTime time.Time

	silenceMu      sync.Mutex
	silenceCounter int
}

func NewElevenLabsTTS(configMap map[string]any) (actions.Connector, error) {
	var cfg ElevenLabsConfig
	if b, err := json.Marshal(configMap); err == nil {
		_ = json.Unmarshal(b, &cfg)
	}
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("speak/elevenlabs_tts: api_key required")
	}
	if cfg.VoiceID == "" {
		cfg.VoiceID = defaultVoiceID
	}
	if cfg.ModelID == "" {
		cfg.ModelID = defaultModelID
	}
	if cfg.OutputFormat == "" {
		cfg.OutputFormat = defaultOutputFormat
	}
	if cfg.Rate == 0 {
		cfg.Rate = rateFromFormat(cfg.OutputFormat)
	}

	ctx, cancel := context.WithCancel(context.Background())
	log, _ := zap.NewProduction()

	c := &ElevenLabsConnector{
		cfg:           cfg,
		log:           log,
		ctx:           ctx,
		cancel:        cancel,
		queue:         make(chan TTSRequest, ttsQueueDepth),
		lastAudioTime: time.Now(),
	}
	c.silenceAudio = silenceBytes(cfg.Rate, 50)

	c.wg.Add(2)
	go c.processAudio()
	go c.keepalive()

	return c, nil
}

// rateFromFormat extracts the sample rate from strings like "pcm_16000".
func rateFromFormat(format string) int {
	parts := strings.Split(format, "_")
	if len(parts) >= 2 {
		if r, err := strconv.Atoi(parts[len(parts)-1]); err == nil {
			return r
		}
	}
	return defaultRate
}

// silenceBytes generates silent 16-bit PCM audio for the given duration.
func silenceBytes(rate, ms int) []byte {
	return make([]byte, rate*ms/1000*2)
}

// Connect enqueues the spoken text without blocking the action pipeline.
func (e *ElevenLabsConnector) Connect(_ context.Context, input actions.Input) (actions.Output, error) {
	arguments, ok := input.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("speak/elevenlabs_tts: unexpected input type %T", input)
	}
	text, _ := arguments["action"].(string)
	if text == "" {
		return nil, nil
	}

	e.silenceMu.Lock()
	if e.cfg.SilenceRate > 0 && e.silenceCounter < e.cfg.SilenceRate {
		e.silenceCounter++
		e.silenceMu.Unlock()
		e.log.Info("speak/elevenlabs_tts: skipping (silence_rate)", zap.Int("counter", e.silenceCounter))
		return nil, nil
	}
	e.silenceCounter = 0
	e.silenceMu.Unlock()

	select {
	case e.queue <- TTSRequest{text: text, voiceID: e.cfg.VoiceID}:
	default:
		e.log.Warn("speak/elevenlabs_tts: queue full, dropping", zap.String("text", text))
	}
	return nil, nil
}

// processAudio serialises synthesis so only one utterance plays at a time.
func (e *ElevenLabsConnector) processAudio() {
	defer e.wg.Done()
	for {
		select {
		case <-e.ctx.Done():
			return
		case req, ok := <-e.queue:
			if !ok {
				return
			}
			if !e.initFFPlay() {
				e.log.Error("speak/elevenlabs_tts: ffplay unavailable")
				continue
			}
			// Silence prefix warms up Bluetooth link before audio starts.
			e.streamChunk(silenceBytes(e.cfg.Rate, 10))

			providers.Speaking.Store(true)
			if err := e.synthesize(req); err != nil && e.ctx.Err() == nil {
				e.log.Error("speak/elevenlabs_tts: synthesis failed", zap.Error(err))
			}
			e.finishPlayback()
			providers.Speaking.Store(false)
		}
	}
}

// keepalive plays silence every 60 s to prevent Bluetooth disconnection.
func (e *ElevenLabsConnector) keepalive() {
	defer e.wg.Done()
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-e.ctx.Done():
			return
		case <-ticker.C:
			e.lastAudioMu.Lock()
			elapsed := time.Since(e.lastAudioTime)
			e.lastAudioMu.Unlock()
			if elapsed >= keepaliveInterval {
				if e.initFFPlay() {
					e.streamChunk(e.silenceAudio)
				}
				e.lastAudioMu.Lock()
				e.lastAudioTime = time.Now()
				e.lastAudioMu.Unlock()
			}
		}
	}
}

// initFFPlay starts a new ffplay process if one is not already running.
func (e *ElevenLabsConnector) initFFPlay() bool {
	e.ffplayMu.Lock()
	defer e.ffplayMu.Unlock()

	if e.ffplay != nil && e.ffplay.ProcessState == nil {
		return true // process still running
	}

	args := []string{}
	if strings.Contains(e.cfg.OutputFormat, "pcm") {
		args = append(args, "-f", "s16le", "-ar", strconv.Itoa(e.cfg.Rate))
	}
	args = append(args, "-nodisp", "-autoexit", "-")

	cmd := exec.CommandContext(e.ctx, "ffplay", args...)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		e.log.Error("speak/elevenlabs_tts: ffplay stdin pipe", zap.Error(err))
		return false
	}

	if err := cmd.Start(); err != nil {
		e.log.Error("speak/elevenlabs_tts: ffplay start", zap.Error(err))
		return false
	}

	e.ffplay = cmd
	e.ffplayIn = stdin
	e.log.Debug("speak/elevenlabs_tts: ffplay started")
	return true
}

// streamChunk writes a chunk of audio to the ffplay stdin pipe.
func (e *ElevenLabsConnector) streamChunk(chunk []byte) {
	e.ffplayMu.Lock()
	defer e.ffplayMu.Unlock()

	if e.ffplayIn == nil {
		return
	}

	if _, err := e.ffplayIn.Write(chunk); err != nil {
		e.log.Warn("speak/elevenlabs_tts: ffplay write failed, reinitializing", zap.Error(err))
		e.cleanupFFPlayLocked()
		return
	}

	e.lastAudioMu.Lock()
	e.lastAudioTime = time.Now()
	e.lastAudioMu.Unlock()
}

// finishPlayback closes stdin and waits for ffplay to drain and exit.
func (e *ElevenLabsConnector) finishPlayback() {
	e.ffplayMu.Lock()
	defer e.ffplayMu.Unlock()

	if e.ffplay == nil {
		return
	}

	if e.ffplayIn != nil {
		_ = e.ffplayIn.Close()
		e.ffplayIn = nil
	}

	waitDone := make(chan struct{})
	go func() {
		_ = e.ffplay.Wait()
		close(waitDone)
	}()

	select {
	case <-waitDone:
		e.log.Debug("speak/elevenlabs_tts: ffplay finished")
	case <-time.After(10 * time.Second):
		e.log.Warn("speak/elevenlabs_tts: ffplay timeout, killing")
		_ = e.ffplay.Process.Kill()
		<-waitDone
	}

	e.ffplay = nil
}

// cleanupFFPlayLocked force-terminates ffplay. Must be called with ffplayMu held.
func (e *ElevenLabsConnector) cleanupFFPlayLocked() {
	if e.ffplayIn != nil {
		_ = e.ffplayIn.Close()
		e.ffplayIn = nil
	}
	if e.ffplay != nil && e.ffplay.Process != nil {
		_ = e.ffplay.Process.Kill()
		_ = e.ffplay.Wait()
		e.ffplay = nil
	}
}

// synthesize posts the text to the OpenAI-compatible speech endpoint and streams
// the response body in 1024-byte chunks to ffplay.
func (e *ElevenLabsConnector) synthesize(req TTSRequest) error {
	body := map[string]any{
		"model":           e.cfg.ModelID,
		"voice":           req.voiceID,
		"response_format": e.cfg.OutputFormat,
		"input":           req.text,
	}

	if e.cfg.ElevenLabsAPIKey != "" {
		body["elevenlabs_api_key"] = e.cfg.ElevenLabsAPIKey
	}

	bodyBytes, err := json.Marshal(body)
	if err != nil {
		return fmt.Errorf("marshal: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(e.ctx, http.MethodPost,
		defaultTTSBaseURL+"/audio/speech", bytes.NewReader(bodyBytes))
	if err != nil {
		return fmt.Errorf("request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("x-api-key", e.cfg.APIKey)

	resp, err := httpclient.Default().Do(httpReq)
	if err != nil {
		return fmt.Errorf("http: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("api %d: %s", resp.StatusCode, b)
	}

	buf := make([]byte, 1024)
	for {
		n, err := resp.Body.Read(buf)
		if n > 0 {
			e.streamChunk(buf[:n])
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("stream: %w", err)
		}
	}
	return nil
}

func (e *ElevenLabsConnector) Tick(ctx context.Context) {
	select {
	case <-ctx.Done():
	case <-time.After(60 * time.Second):
	}
}

func (e *ElevenLabsConnector) Stop() {
	e.cancel()
	e.wg.Wait()
	e.cleanupFFPlayLocked()
}
