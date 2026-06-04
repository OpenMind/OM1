package inputs

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gordonklaus/portaudio"
	"go.uber.org/zap"

	"github.com/openmind/om1/internal/inputs"
	"github.com/openmind/om1/internal/metrics"
	"github.com/openmind/om1/internal/providers"
	"github.com/openmind/om1/internal/util"
)

func init() {
	inputs.Register("ParallelASRInput", NewParallelASR)
}

// defaultParallelDedupWindow is how long, after one provider's transcript is
// accepted, transcripts from the *other* providers are suppressed as duplicates
// of the same utterance. The winning provider is never suppressed.
const defaultParallelDedupWindow = 3 * time.Second

// ParallelASRProviderConfig configures one ASR provider within a parallel sensor.
type ParallelASRProviderConfig struct {
	Model                string   `json:"model"`    // "riva" | "google" | "elevenlabs"
	APIKey               string   `json:"api_key"`  // required for google/elevenlabs
	APIVersion           string   `json:"api_version"`
	BaseURL              string   `json:"base_url"` // override WS endpoint
	Language             string   `json:"language"`
	AlternativeLanguages []string `json:"alternative_languages"` // google v1 only
	DelayMS              int      `json:"delay_ms"`              // artificially delay this provider's transcripts (handicap for testing/biasing the race)
}

// ParallelASRConfig configures a sensor that captures audio once and streams it to
// multiple ASR providers concurrently, surfacing whichever transcript arrives first.
type ParallelASRConfig struct {
	Source             string                      `json:"source"`               // "mic" (default) | "rtsp"
	APIKey             string                      `json:"api_key"`              // default key for providers that omit one (e.g. the global config key)
	Rate               int                         `json:"rate"`                 // sample rate Hz
	Chunk              int                         `json:"chunk"`                // frames/samples per chunk
	MicDeviceIndex     int                         `json:"microphone_device_id"` // -1 = default (mic source)
	RTSPURL            string                      `json:"rtsp_url"`             // rtsp source
	DedupWindowMS      int                         `json:"dedup_window_ms"`      // first-wins suppression window (default 1000)
	EnableTTSInterrupt bool                        `json:"enable_tts_interrupt"`
	Providers          []ParallelASRProviderConfig `json:"providers"`
}

// ParallelASRSensor captures audio from a single source (mic or RTSP), fans the PCM
// out to every configured provider's websocket stream, and forwards the first
// transcript per utterance to the cortex loop while suppressing the slower duplicates.
type ParallelASRSensor struct {
	*asrAggregator

	cfg     ParallelASRConfig
	streams []*transcriberStream

	// mic source
	paStream   *portaudio.Stream
	audioChunk []int16

	// first-wins dedup state
	dedupWindow time.Duration
	dmu         sync.Mutex
	lastModel   string
	lastTime    time.Time
}

// NewParallelASR constructs a ParallelASRSensor with the given configuration.
func NewParallelASR(configMap map[string]any) (inputs.Sensor, error) {
	cfg := ParallelASRConfig{MicDeviceIndex: -1}
	if b, err := json.Marshal(configMap); err == nil {
		_ = json.Unmarshal(b, &cfg)
	}

	cfg.Source = strings.TrimSpace(strings.ToLower(cfg.Source))
	if cfg.Source == "" {
		cfg.Source = "mic"
	}
	if cfg.Source != "mic" && cfg.Source != "rtsp" {
		return nil, fmt.Errorf("ParallelASRInput: source must be \"mic\" or \"rtsp\", got %q", cfg.Source)
	}

	if len(cfg.Providers) == 0 {
		return nil, fmt.Errorf("ParallelASRInput: at least one provider is required")
	}

	if cfg.Rate == 0 {
		if cfg.Source == "mic" {
			cfg.Rate = 48000
		} else {
			cfg.Rate = 16000
		}
	}
	if cfg.Chunk == 0 {
		if cfg.Source == "mic" {
			cfg.Chunk = 4800
		} else {
			cfg.Chunk = 1600
		}
	}

	dedupWindow := defaultParallelDedupWindow
	if cfg.DedupWindowMS > 0 {
		dedupWindow = time.Duration(cfg.DedupWindowMS) * time.Millisecond
	}

	s := &ParallelASRSensor{
		asrAggregator: newAggregator("ParallelASRInput"),
		cfg:           cfg,
		dedupWindow:   dedupWindow,
	}
	if cfg.Source == "mic" {
		s.audioChunk = make([]int16, cfg.Chunk)
	}

	for i, p := range cfg.Providers {
		streamCfg, err := s.buildStreamConfig(p)
		if err != nil {
			return nil, fmt.Errorf("ParallelASRInput: provider %d: %w", i, err)
		}
		delay := time.Duration(p.DelayMS) * time.Millisecond
		s.streams = append(s.streams, newTranscriberStream(streamCfg, s.log, s.deliverWithDelay(delay)))
	}

	s.log.Info("ParallelASRInput: configured",
		zap.String("source", cfg.Source),
		zap.Int("rate", cfg.Rate),
		zap.Int("chunk", cfg.Chunk),
		zap.Int("providers", len(s.streams)),
		zap.Duration("dedup_window", dedupWindow),
	)

	return s, nil
}

// buildStreamConfig resolves one provider entry into a transcriberStream config,
// reusing each vendor's existing config resolution (endpoints, language mapping).
func (s *ParallelASRSensor) buildStreamConfig(p ParallelASRProviderConfig) (asrCommonConfig, error) {
	model := strings.TrimSpace(strings.ToLower(p.Model))
	name := fmt.Sprintf("ParallelASRInput[%s]", model)

	// Providers inherit the sensor-level api_key
	apiKey := p.APIKey
	if apiKey == "" {
		apiKey = s.cfg.APIKey
	}

	switch model {
	case "riva":
		return resolveRivaASRConfig(rivaASRParams{
			name:               name,
			baseURL:            p.BaseURL,
			rate:               s.cfg.Rate,
			language:           p.Language,
			enableTTSInterrupt: s.cfg.EnableTTSInterrupt,
		}), nil
	case "google":
		if apiKey == "" {
			return asrCommonConfig{}, fmt.Errorf("google provider requires api_key")
		}
		return resolveGoogleASRConfig(googleASRParams{
			name:                 name,
			apiKey:               apiKey,
			apiVersion:           p.APIVersion,
			baseURL:              p.BaseURL,
			rate:                 s.cfg.Rate,
			language:             p.Language,
			alternativeLanguages: p.AlternativeLanguages,
			enableTTSInterrupt:   s.cfg.EnableTTSInterrupt,
		}), nil
	case "elevenlabs":
		if apiKey == "" {
			return asrCommonConfig{}, fmt.Errorf("elevenlabs provider requires api_key")
		}
		return resolveElevenLabsASRConfig(elevenlabsASRParams{
			name:               name,
			apiKey:             apiKey,
			baseURL:            p.BaseURL,
			rate:               s.cfg.Rate,
			language:           p.Language,
			enableTTSInterrupt: s.cfg.EnableTTSInterrupt,
		}), nil
	default:
		return asrCommonConfig{}, fmt.Errorf("unknown provider model %q (want riva, google, or elevenlabs)", p.Model)
	}
}

// deliverWithDelay returns an onTranscript handler that optionally postpones a
// provider's transcripts by delay before they enter first-wins dedup. A delay
// handicaps an otherwise-dominant provider so slower ones can win — useful for
// testing and for biasing the race toward a preferred provider. The delivery is
// scheduled (not slept inline) so the websocket read loop is never blocked.
func (s *ParallelASRSensor) deliverWithDelay(delay time.Duration) func(model, text string) {
	if delay <= 0 {
		return s.deliver
	}
	return func(model, text string) {
		time.AfterFunc(delay, func() { s.deliver(model, text) })
	}
}

// deliver applies first-wins dedup: the first provider to transcribe an utterance
// wins and is forwarded.
func (s *ParallelASRSensor) deliver(model, text string) {
	s.dmu.Lock()
	now := time.Now()
	if model != s.lastModel && now.Sub(s.lastTime) < s.dedupWindow {
		winner := s.lastModel
		s.dmu.Unlock()
		metrics.ASRParallelTranscripts.WithLabelValues(model, "suppressed").Inc()
		s.log.Info("ParallelASRInput: lost race, suppressed duplicate",
			zap.String("loser", model),
			zap.String("winner", winner),
			zap.String("text", text),
		)
		return
	}
	s.lastModel = model
	s.lastTime = now
	s.dmu.Unlock()

	metrics.ASRParallelTranscripts.WithLabelValues(model, "won").Inc()
	s.log.Info("ParallelASRInput: winner", zap.String("winner", model), zap.String("text", text))
	s.pushTranscript(text)
}

// Listen starts the sensor: it connects every provider stream, begins capture from
// the configured source, and forwards merged transcripts until ctx is cancelled.
func (s *ParallelASRSensor) Listen(ctx context.Context) (<-chan any, error) {
	out := make(chan any)
	go func() {
		defer close(out)
		defer s.Stop()

		if s.cfg.Source == "mic" {
			if err := providers.PortAudio.Acquire(); err != nil {
				s.log.Error("ParallelASRInput: portaudio init failed", zap.Error(err))
				return
			}
		}

		s.connectStreams()

		for _, st := range s.streams {
			go st.statsLoop(ctx)
		}

		if s.cfg.Source == "mic" {
			if err := s.openMic(ctx); err != nil {
				s.log.Error("ParallelASRInput: mic open failed", zap.Error(err))
				return
			}
		} else {
			s.mu.Lock()
			s.captureDone = make(chan struct{})
			s.mu.Unlock()
			go s.rtspCaptureLoop(ctx)
		}

		s.pollLoop(ctx, out)
	}()
	return out, nil
}

// connectStreams dials every provider websocket concurrently.
func (s *ParallelASRSensor) connectStreams() {
	var wg sync.WaitGroup
	for _, st := range s.streams {
		wg.Add(1)
		go func(st *transcriberStream) {
			defer wg.Done()
			if err := st.connect(); err != nil {
				s.log.Warn("ParallelASRInput: provider ws connect failed, will retry",
					zap.String("model", st.model), zap.Error(err))
			}
		}(st)
	}
	wg.Wait()
}

// sendToAll fans one PCM chunk out to every provider stream. Each stream packages
// the audio with its own header, so the shared chunk is only read, never mutated.
func (s *ParallelASRSensor) sendToAll(pcm []byte) {
	for _, st := range s.streams {
		st.sendChunk(pcm)
	}
}

// Stop signals capture to stop, waits for it to finish, and cleans up resources.
func (s *ParallelASRSensor) Stop() {
	first, captureDone := s.markStopped()
	if !first {
		return
	}

	s.log.Info("ParallelASRInput: stopping sensor")

	s.waitCapture(captureDone)
	for _, st := range s.streams {
		st.closeWS()
	}
	if s.cfg.Source == "mic" {
		providers.PortAudio.Release()
	}
	s.closeZenoh()

	s.log.Info("ParallelASRInput: sensor stopped")
}

// openMic opens the configured microphone stream and starts the capture loop.
func (s *ParallelASRSensor) openMic(ctx context.Context) error {
	var device *portaudio.DeviceInfo
	var err error

	if s.cfg.MicDeviceIndex >= 0 {
		devices, err := portaudio.Devices()
		if err != nil {
			return fmt.Errorf("ParallelASRInput: list devices: %w", err)
		}
		if s.cfg.MicDeviceIndex >= len(devices) {
			return fmt.Errorf("ParallelASRInput: device index %d out of range", s.cfg.MicDeviceIndex)
		}
		device = devices[s.cfg.MicDeviceIndex]
	} else {
		device, err = portaudio.DefaultInputDevice()
		if err != nil {
			return fmt.Errorf("ParallelASRInput: default device: %w", err)
		}
	}

	s.log.Info("ParallelASRInput: microphone",
		zap.String("device", device.Name),
		zap.Int("rate", s.cfg.Rate),
		zap.Int("chunk", s.cfg.Chunk),
		zap.Float64("chunk_ms", float64(s.cfg.Chunk)/float64(s.cfg.Rate)*1000),
	)

	params := portaudio.StreamParameters{
		Input: portaudio.StreamDeviceParameters{
			Device:   device,
			Channels: 1,
			Latency:  device.DefaultHighInputLatency,
		},
		SampleRate:      float64(s.cfg.Rate),
		FramesPerBuffer: s.cfg.Chunk,
	}

	stream, err := portaudio.OpenStream(params, s.audioChunk)
	if err != nil {
		return fmt.Errorf("ParallelASRInput: open stream: %w", err)
	}

	if err := stream.Start(); err != nil {
		_ = stream.Close()
		return fmt.Errorf("ParallelASRInput: start stream: %w", err)
	}

	s.mu.Lock()
	s.paStream = stream
	s.captureDone = make(chan struct{})
	s.mu.Unlock()

	go s.micCaptureLoop(ctx, stream)
	s.log.Info("ParallelASRInput: microphone started")
	return nil
}

// micCaptureLoop reads PCM from the microphone and fans it out to all providers.
func (s *ParallelASRSensor) micCaptureLoop(ctx context.Context, stream *portaudio.Stream) {
	defer func() {
		_ = stream.Stop()
		_ = stream.Close()

		s.mu.Lock()
		if s.paStream == stream {
			s.paStream = nil
		}
		done := s.captureDone
		s.mu.Unlock()

		if done != nil {
			close(done)
		}
	}()

	for {
		select {
		case <-ctx.Done():
			return
		default:
		}

		if err := stream.Read(); err != nil && err.Error() != "Input overflowed" {
			s.log.Warn("ParallelASRInput: read error", zap.Error(err))
		}

		if providers.Speaking.Load() && !s.cfg.EnableTTSInterrupt {
			continue
		}

		pcm := make([]byte, len(s.audioChunk)*2)
		for i, sample := range s.audioChunk {
			binary.LittleEndian.PutUint16(pcm[i*2:], uint16(sample))
		}

		s.sendToAll(pcm)
	}
}

// rtspCaptureLoop runs the RTSP stream and reconnects on failure until ctx is cancelled.
func (s *ParallelASRSensor) rtspCaptureLoop(ctx context.Context) {
	defer func() {
		s.mu.Lock()
		done := s.captureDone
		s.mu.Unlock()
		if done != nil {
			close(done)
		}
	}()

	for {
		if ctx.Err() != nil {
			return
		}

		if err := s.streamRTSP(ctx); err != nil && ctx.Err() == nil {
			s.log.Warn("ParallelASRInput: RTSP audio error", zap.Error(err))
			s.log.Info("ParallelASRInput: reconnecting", zap.Duration("delay", rtspReconnectDelay))
			if !util.Sleep(ctx, rtspReconnectDelay) {
				return
			}
		}
	}
}

// streamRTSP runs ffmpeg to capture PCM audio from the RTSP URL and fans it out to
// all providers until an error occurs or ctx is cancelled.
func (s *ParallelASRSensor) streamRTSP(ctx context.Context) error {
	cmd := exec.CommandContext(ctx, "ffmpeg",
		"-rtsp_transport", "tcp",
		"-i", s.cfg.RTSPURL,
		"-vn",
		"-f", "s16le",
		"-acodec", "pcm_s16le",
		"-ac", "1",
		"-ar", strconv.Itoa(s.cfg.Rate),
		"-loglevel", "error",
		"pipe:1",
	)

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("stdout pipe: %w", err)
	}
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("start ffmpeg: %w", err)
	}
	defer func() {
		if cmd.Process != nil {
			_ = cmd.Process.Kill()
		}
		_ = cmd.Wait()
	}()

	s.log.Info("ParallelASRInput: RTSP audio stream connected", zap.String("rtsp_url", s.cfg.RTSPURL))

	chunkBytes := s.cfg.Chunk * 2 // int16 samples
	buf := make([]byte, chunkBytes)
	for {
		if ctx.Err() != nil {
			return ctx.Err()
		}

		if _, err := io.ReadFull(stdout, buf); err != nil {
			return fmt.Errorf("read pcm: %w", err)
		}

		if providers.Speaking.Load() && !s.cfg.EnableTTSInterrupt {
			continue
		}

		pcm := make([]byte, chunkBytes)
		copy(pcm, buf)
		s.sendToAll(pcm)
	}
}
