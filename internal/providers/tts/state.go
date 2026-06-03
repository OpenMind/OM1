package tts

import "sync/atomic"

// Speaking is set to true while a TTS connector is streaming audio.
// Input sensors (e.g. GoogleASR) check this to suppress capture during playback.
var Speaking atomic.Bool

// Interrupt is set to true by input sensors when user speech is detected during TTS playback.
var Interrupt atomic.Bool
