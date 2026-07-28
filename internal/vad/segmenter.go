package vad

import (
	"encoding/binary"
	"time"
)

// Event is a speech-boundary event detected by a Segmenter.
type Event struct {
	Type string // "speech_start" or "speech_end"
	At   time.Time
}

const (
	EventSpeechStart = "speech_start"
	EventSpeechEnd   = "speech_end"
)

// SegmenterConfig tunes the speech/silence decision.
type SegmenterConfig struct {
	// Threshold is the speech-probability cutoff (default 0.5, Silero's own default).
	Threshold float32
	// MinSilenceDuration is how long the probability must stay below Threshold,
	// after speech, before speech_end fires (default 300ms).
	MinSilenceDuration time.Duration
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

// Segmenter turns a raw mono PCM stream at an arbitrary sample rate into
// speech_start/speech_end events, resampling to 16kHz and running a Model
// over consecutive 512-sample frames. It is not safe for concurrent use.
type Segmenter struct {
	model   *Model
	cfg     SegmenterConfig
	srcRate int

	srcBuf []int16 // unconsumed source-rate samples, used only when srcRate != SampleRate
	srcPos float64 // fractional read position into srcBuf

	frameBuf []float32 // resampled 16kHz samples awaiting a full frame

	inSpeech     bool
	haveSilence  bool
	silenceSince time.Time
}

// NewSegmenter builds a Segmenter reading PCM captured at srcRate and running
// inference on model. model may be shared across Segmenters only if calls
// are externally serialized; typically each stream owns its own Model.
func NewSegmenter(model *Model, srcRate int, cfg SegmenterConfig) *Segmenter {
	return &Segmenter{
		model:   model,
		cfg:     cfg.withDefaults(),
		srcRate: srcRate,
	}
}

// Feed processes one chunk of little-endian int16 mono PCM. t is the
// wall-clock time the whole chunk was captured/observed; all frames drained
// from this chunk are timestamped with it, which is accurate to within one
// chunk's duration -- adequate for measuring second-scale ASR latency.
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

// observe applies the threshold+hangover state machine to one frame's speech
// probability, returning a boundary event if the frame completes one.
// speech_end is timestamped at the moment silence actually began, not the
// (later) moment it was confirmed -- that's the true stop-speaking instant.
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
// carrying fractional position and unconsumed tail samples across calls so
// consecutive chunks resample continuously.
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

	// srcPos can overshoot the buffer (e.g. an exact integer ratio leaves no
	// fractional part to anchor the next sample): cap the trim at the
	// buffer's length, but keep the overshoot in srcPos so it carries over
	// as a skip into the samples the next call appends.
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
