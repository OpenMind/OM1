package memory

import (
	"archive/tar"
	"compress/gzip"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
)

func TestUploader_CheckEligibility_NoDaily(t *testing.T) {
	root := t.TempDir()
	u := NewUploader(root, "test-key", zap.NewNop())
	assert.False(t, u.CheckEligibility(), "should be false with no daily dir")
}

func TestUploader_CheckEligibility_BelowThreshold(t *testing.T) {
	root := t.TempDir()
	dailyDir := filepath.Join(root, "daily")
	require.NoError(t, os.MkdirAll(dailyDir, 0o755))

	// Write 5 sections — below threshold of 100.
	content := ""
	for i := 0; i < 5; i++ {
		content += "## 10:00:0" + string(rune('0'+i)) + "\nSome interaction\n\n"
	}
	require.NoError(t, os.WriteFile(
		filepath.Join(dailyDir, time.Now().Format("2006-01-02")+".md"),
		[]byte(content), 0o644,
	))

	u := NewUploader(root, "test-key", zap.NewNop())
	assert.False(t, u.CheckEligibility(), "5 sections should be below threshold")
}

func TestUploader_CheckEligibility_AboveThreshold(t *testing.T) {
	root := t.TempDir()
	dailyDir := filepath.Join(root, "daily")
	require.NoError(t, os.MkdirAll(dailyDir, 0o755))

	// Write 101 sections — above threshold.
	content := ""
	for i := 0; i < 101; i++ {
		h := i / 3600
		m := (i % 3600) / 60
		s := i % 60
		content += "## " + time.Date(2026, 1, 1, h, m, s, 0, time.UTC).Format("15:04:05") + "\nSome interaction\n\n"
	}
	require.NoError(t, os.WriteFile(
		filepath.Join(dailyDir, time.Now().Format("2006-01-02")+".md"),
		[]byte(content), 0o644,
	))

	u := NewUploader(root, "test-key", zap.NewNop())
	assert.True(t, u.CheckEligibility(), "101 sections should exceed threshold")
}

func TestUploader_CheckEligibility_RespectsMarker(t *testing.T) {
	root := t.TempDir()
	dailyDir := filepath.Join(root, "daily")
	require.NoError(t, os.MkdirAll(dailyDir, 0o755))

	now := time.Now()
	content := ""
	// 50 sections before marker time, 60 after.
	for i := 0; i < 110; i++ {
		ts := now.Add(time.Duration(i-50) * time.Minute)
		content += "## " + ts.Format("15:04:05") + "\nSome interaction\n\n"
	}
	require.NoError(t, os.WriteFile(
		filepath.Join(dailyDir, now.Format("2006-01-02")+".md"),
		[]byte(content), 0o644,
	))

	// Set marker to "now" — only 60 sections after marker.
	u := NewUploader(root, "test-key", zap.NewNop())
	require.NoError(t, os.WriteFile(u.markerFile, []byte(now.Format("2006-01-02 15:04")), 0o644))

	assert.False(t, u.CheckEligibility(), "60 sections after marker should be below threshold")
}

func TestUploader_CreateTarGz_ExcludesIndex(t *testing.T) {
	root := t.TempDir()
	// Create memory structure.
	require.NoError(t, os.MkdirAll(filepath.Join(root, "daily"), 0o755))
	require.NoError(t, os.MkdirAll(filepath.Join(root, "index"), 0o755))
	require.NoError(t, os.WriteFile(filepath.Join(root, "daily", "2026-06-12.md"), []byte("test"), 0o644))
	require.NoError(t, os.WriteFile(filepath.Join(root, "index", "index.graph"), []byte("binary"), 0o644))
	require.NoError(t, os.WriteFile(filepath.Join(root, "index", "index.meta.json"), []byte("{}"), 0o644))

	u := NewUploader(root, "test-key", zap.NewNop())

	tmpFile, err := os.CreateTemp("", "test-tar-*.tar.gz")
	require.NoError(t, err)
	defer os.Remove(tmpFile.Name())

	require.NoError(t, u.createTarGz(tmpFile))
	_ = tmpFile.Close()

	// Verify archive contents.
	entries := listTarEntries(t, tmpFile.Name())
	assert.Contains(t, entries, "memory/daily/2026-06-12.md")
	assert.NotContains(t, entries, "memory/index/index.graph")
	assert.NotContains(t, entries, "memory/index/index.meta.json")
}

func TestUploader_MarkerRoundtrip(t *testing.T) {
	root := t.TempDir()
	u := NewUploader(root, "test-key", zap.NewNop())

	assert.Nil(t, u.readMarker(), "no marker file should return nil")

	u.writeMarker()
	m := u.readMarker()
	require.NotNil(t, m)
	assert.WithinDuration(t, time.Now(), *m, 2*time.Minute)
}

// listTarEntries returns all file names in a tar.gz archive.
func listTarEntries(t *testing.T, path string) []string {
	t.Helper()
	f, err := os.Open(path)
	require.NoError(t, err)
	defer func() { _ = f.Close() }()

	gz, err := gzip.NewReader(f)
	require.NoError(t, err)
	defer func() { _ = gz.Close() }()

	tr := tar.NewReader(gz)
	var names []string
	for {
		h, err := tr.Next()
		if err != nil {
			break
		}
		names = append(names, h.Name)
	}
	return names
}
