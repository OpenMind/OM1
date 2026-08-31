package vlm

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMemoryOptionsDefaults(t *testing.T) {
	cfg := MemoryOptions{}.Config()
	assert.Equal(t, defaultMemoryFrames, cfg.MaxFrames)
	assert.Equal(t, defaultMemoryTexts, cfg.MaxTexts)
	assert.Equal(t, defaultMemoryCapacity, cfg.Capacity)
	assert.Equal(t, 2*time.Second, cfg.MinInterval)
}

func TestMemoryOptionsExplicitZeroDisablesFrames(t *testing.T) {
	var opts MemoryOptions
	require.NoError(t, json.Unmarshal([]byte(`{"memory_frames":0,"memory_texts":0}`), &opts))

	cfg := opts.Config()
	assert.Equal(t, 0, cfg.MaxFrames)
	assert.Equal(t, 0, cfg.MaxTexts)
	assert.False(t, NewMemory(cfg).Enabled())
}

func TestMemoryOptionsFromJSON(t *testing.T) {
	var opts MemoryOptions
	require.NoError(t, json.Unmarshal([]byte(`{
		"memory_frames": 4,
		"memory_texts": 6,
		"memory_interval_sec": 0.5,
		"memory_capacity": 20,
		"memory_sampling": "recency"
	}`), &opts))

	cfg := opts.Config()
	assert.Equal(t, 4, cfg.MaxFrames)
	assert.Equal(t, 6, cfg.MaxTexts)
	assert.Equal(t, 500*time.Millisecond, cfg.MinInterval)
	assert.Equal(t, 20, cfg.Capacity)
	assert.Equal(t, SamplingRecency, cfg.Sampling)
}

func TestNilMemoryIsSafe(t *testing.T) {
	var m *Memory
	assert.False(t, m.Enabled())
	assert.True(t, m.History(time.Now()).Empty())
	m.Add(time.Now(), "frame", "text")
}

func TestMemoryRespectsMinInterval(t *testing.T) {
	m := NewMemory(MemoryConfig{MaxFrames: 5, MaxTexts: 5, Capacity: 10, MinInterval: time.Second})
	base := time.Unix(100, 0)

	m.Add(base, "a", "first")
	m.Add(base.Add(200*time.Millisecond), "b", "too soon")
	m.Add(base.Add(1200*time.Millisecond), "c", "second")

	hist := m.History(base.Add(2 * time.Second))
	require.Len(t, hist.Frames, 2)
	assert.Equal(t, "first", hist.Frames[0].Description)
	assert.Equal(t, "second", hist.Frames[1].Description)
	assert.Equal(t, 2*time.Second, hist.Frames[0].Age)
	assert.Equal(t, 800*time.Millisecond, hist.Frames[1].Age)
}

func TestMemorySkipsEmptyDescriptions(t *testing.T) {
	m := NewMemory(MemoryConfig{MaxFrames: 3, MaxTexts: 3, Capacity: 5})
	m.Add(time.Unix(1, 0), "a", "")
	assert.True(t, m.History(time.Unix(2, 0)).Empty())
}

func TestMemoryEvictsOldestBeyondCapacity(t *testing.T) {
	m := NewMemory(MemoryConfig{MaxFrames: 10, MaxTexts: 0, Capacity: 3})
	base := time.Unix(0, 0)
	for i := 0; i < 6; i++ {
		m.Add(base.Add(time.Duration(i)*time.Second), "f", string(rune('a'+i)))
	}

	hist := m.History(base.Add(10 * time.Second))
	require.Len(t, hist.Frames, 3)
	assert.Equal(t, "d", hist.Frames[0].Description)
	assert.Equal(t, "f", hist.Frames[2].Description)
}

func TestMemoryTextCacheExcludesSampledFrames(t *testing.T) {
	m := NewMemory(MemoryConfig{MaxFrames: 2, MaxTexts: 3, Capacity: 10})
	base := time.Unix(0, 0)
	for i := 0; i < 5; i++ {
		m.Add(base.Add(time.Duration(i)*time.Second), "f", string(rune('a'+i)))
	}

	// Frames sampled at the ends of the window; the recent tail that was not
	// sampled as an image comes back as text only.
	hist := m.History(base.Add(5 * time.Second))
	require.Len(t, hist.Frames, 2)
	assert.Equal(t, "a", hist.Frames[0].Description)
	assert.Equal(t, "e", hist.Frames[1].Description)

	require.Len(t, hist.Texts, 2)
	assert.Equal(t, "c", hist.Texts[0].Description)
	assert.Equal(t, "d", hist.Texts[1].Description)
	for _, s := range hist.Texts {
		assert.Empty(t, s.JPEGBase64, "text cache must not carry image data")
	}
}

func TestSampleIndices(t *testing.T) {
	tests := []struct {
		name     string
		n, k     int
		sampling string
		want     []int
	}{
		{"none", 0, 3, SamplingUniform, nil},
		{"disabled", 5, 0, SamplingUniform, nil},
		{"all when k exceeds n", 3, 5, SamplingUniform, []int{0, 1, 2}},
		{"newest when k is one", 5, 1, SamplingUniform, []int{4}},
		{"uniform keeps both ends", 10, 3, SamplingUniform, []int{0, 5, 9}},
		{"uniform spreads evenly", 9, 5, SamplingUniform, []int{0, 2, 4, 6, 8}},
		{"recency clusters at newest", 10, 4, SamplingRecency, []int{0, 7, 8, 9}},
		{"recency keeps oldest", 10, 2, SamplingRecency, []int{0, 9}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := sampleIndices(tc.n, tc.k, tc.sampling)
			assert.Equal(t, tc.want, got)
			assert.LessOrEqual(t, len(got), tc.k)
		})
	}
}

func TestBuildContentSingleFrame(t *testing.T) {
	content := buildContent("describe", "IMG", History{})
	require.Len(t, content, 2)
	assert.Equal(t, "describe", content[0]["text"])
	assert.Equal(t, "image_url", content[1]["type"])
}

func TestBuildContentOmitsImageWhenCaptureFailed(t *testing.T) {
	content := buildContent("describe", "", History{})
	require.Len(t, content, 1)
	assert.Equal(t, "describe", content[0]["text"])
}

func TestBuildContentInterleavesHistory(t *testing.T) {
	hist := History{
		Frames: []HistoryStep{
			{Age: 6 * time.Second, JPEGBase64: "OLD", Description: "an empty hallway"},
			{Age: 2 * time.Second, JPEGBase64: "MID", Description: "a person approaching"},
		},
		Texts: []HistoryStep{{Age: time.Second, Description: "the person waved"}},
	}

	content := buildContent("describe", "NOW", hist)

	// prompt, header, (text+image)x2, text cache, current header, image, footer
	require.Len(t, content, 10)
	assert.Equal(t, "describe", content[0]["text"])
	assert.Equal(t, historyHeader, content[1]["text"])
	assert.Contains(t, content[2]["text"], "an empty hallway")
	assert.Contains(t, content[2]["text"], "6.0s ago")
	assert.Contains(t, imageURL(t, content[3]), "OLD")
	assert.Contains(t, content[4]["text"], "a person approaching")
	assert.Contains(t, imageURL(t, content[5]), "MID")
	assert.Contains(t, content[6]["text"], textCacheHeader)
	assert.Contains(t, content[6]["text"], "the person waved")
	assert.Equal(t, currentHeader, content[7]["text"])
	assert.Contains(t, imageURL(t, content[8]), "NOW")

	// The instruction closes the message so the model answers for now, not then.
	assert.Equal(t, historyFooter, content[9]["text"])
}

func imageURL(t *testing.T, part map[string]any) string {
	t.Helper()
	require.Equal(t, "image_url", part["type"])
	img, ok := part["image_url"].(map[string]any)
	require.True(t, ok)
	url, ok := img["url"].(string)
	require.True(t, ok)
	return url
}

func TestFormatAge(t *testing.T) {
	assert.Equal(t, "0.0s", formatAge(0))
	assert.Equal(t, "1.5s", formatAge(1500*time.Millisecond))
	assert.Equal(t, "1m2s", formatAge(62*time.Second))
}
