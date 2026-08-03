package vad

import (
	"encoding/binary"
	"time"
)

// Event is a speech-boundary event detected by a Segmenter.
type Event struct {
	Type string //"speech_start" or "speech_end"
	At   time.Time
}

const (
	EventSpeechStart = "speech_start"
	EventSpeechEnd   = "speech_end"
)

// SegmenterConfig tunes the speech/silence decision.
type SegmenterConfig struct {
	Threshold          float32
	MinSilenceDuration time.Duration
}

// Inferer runs one Silero VAD frame-inference step over FrameSamples of new
// mono audio, returning that frame's speech probability. Model (local ONNX)
// and RemoteModel (GPU service over HTTP) both implement it, so a Segmenter
// can run against either backend interchangeably.
type Inferer interface {
	Infer(frame []float32) (float32, error)
}

// Backend is the full lifecycle of a VAD model as owned by a caller that
// constructs and eventually releases it directly (unlike Segmenter, which
// only ever needs Infer): it adds Close so the local (ONNX) and remote
// (HTTP) backends can be released the same way regardless of which one is
// configured. Model and RemoteModel both implement it.
type Backend interface {
	Inferer
	Close() error
}

func (c SegmenterConfig) withDefaults() SegmenterConfig {
	if c.Threshold <= 0 {
		c.Threshold = 0.5
	}
	if c.MinSilenceDuration <= 0 {
		c.MinSilenceDuration = 300 * time.Millisecond
	}
	return c
}

// Segmenter turns a raw mono PCM stream into speech_start/speech_end events
type Segmenter struct {
	model   Inferer
	cfg     SegmenterConfig
	srcRate int

	srcBuf []int16
	srcPos float64

	frameBuf []float32

	inSpeech     bool
	haveSilence  bool
	silenceSince time.Time
}

// NewSegmenter builds a Segmenter reading PCM
func NewSegmenter(model Inferer, srcRate int, cfg SegmenterConfig) *Segmenter {
	return &Segmenter{
		model:   model,
		cfg:     cfg.withDefaults(),
		srcRate: srcRate,
	}
}

// Feed processes one chunk of little-endian int16 mono PCM.
func (s *Segmenter) Feed(pcm []byte, t time.Time) []Event {
	s.frameBuf = append(s.frameBuf, s.resample(bytesToInt16(pcm))...)

	var events []Event
	for len(s.frameBuf) >= FrameSamples {
		frame := s.frameBuf[:FrameSamples]
		s.frameBuf = append([]float32(nil), s.frameBuf[FrameSamples:]...)

		prob, err := s.model.Infer(frame)
		if err != nil {
			continue
		}
		if ev, ok := s.observe(prob, t); ok {
			events = append(events, ev)
		}
	}
	return events
}

// apply the threshold+hangover state machine to one frame's speech probability
func (s *Segmenter) observe(prob float32, t time.Time) (Event, bool) {
	if prob >= s.cfg.Threshold {
		s.haveSilence = false
		if !s.inSpeech {
			s.inSpeech = true
			return Event{Type: EventSpeechStart, At: t}, true
		}
		return Event{}, false
	}

	if !s.inSpeech {
		return Event{}, false
	}

	if !s.haveSilence {
		s.haveSilence = true
		s.silenceSince = t
		return Event{}, false
	}

	if t.Sub(s.silenceSince) >= s.cfg.MinSilenceDuration {
		s.inSpeech = false
		s.haveSilence = false
		return Event{Type: EventSpeechEnd, At: s.silenceSince}, true
	}
	return Event{}, false
}

// resample converts new source-rate samples to 16kHz float32 in [-1, 1],
func (s *Segmenter) resample(newSamples []int16) []float32 {
	if s.srcRate == SampleRate {
		out := make([]float32, len(newSamples))
		for i, v := range newSamples {
			out[i] = float32(v) / 32768.0
		}
		return out
	}

	s.srcBuf = append(s.srcBuf, newSamples...)
	ratio := float64(s.srcRate) / float64(SampleRate)

	var out []float32
	for {
		i0 := int(s.srcPos)
		i1 := i0 + 1
		if i1 >= len(s.srcBuf) {
			break
		}
		frac := s.srcPos - float64(i0)
		v := float64(s.srcBuf[i0])*(1-frac) + float64(s.srcBuf[i1])*frac
		out = append(out, float32(v/32768.0))
		s.srcPos += ratio
	}

	trim := int(s.srcPos)
	if trim <= 0 {
		return out
	}
	if trim > len(s.srcBuf) {
		trim = len(s.srcBuf)
	}
	s.srcBuf = append([]int16(nil), s.srcBuf[trim:]...)
	s.srcPos -= float64(trim)
	return out
}

func bytesToInt16(pcm []byte) []int16 {
	n := len(pcm) / 2
	out := make([]int16, n)
	for i := 0; i < n; i++ {
		out[i] = int16(binary.LittleEndian.Uint16(pcm[i*2:]))
	}
	return out
}
