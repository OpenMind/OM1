package memory

import (
	"archive/tar"
	"compress/gzip"
	"context"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/httpclient"
)

const (
	uploadChunkThreshold = 10
	defaultUploadURL     = "http://localhost:3001/api/core/memory/upload"
	uploadTimeout        = 60 * time.Second
)

// Uploader periodically uploads memory snapshots to S3 via openmind-api.
type Uploader struct {
	memoryRoot string
	dailyDir   string
	markerFile string
	apiKey     string
	uploadURL  string
	client     *http.Client
	log        *zap.Logger

	mu      sync.Mutex
	running bool
}

// NewUploader creates an Uploader.
func NewUploader(memoryRoot, apiKey string, log *zap.Logger) *Uploader {
	return &Uploader{
		memoryRoot: memoryRoot,
		dailyDir:   filepath.Join(memoryRoot, "daily"),
		markerFile: filepath.Join(memoryRoot, ".last_upload"),
		apiKey:     apiKey,
		uploadURL:  defaultUploadURL,
		client:     httpclient.Default(),
		log:        log,
	}
}

// CheckEligibility returns true if new interaction sections since the last upload exceed uploadChunkThreshold.
func (u *Uploader) CheckEligibility() bool {
	u.mu.Lock()
	running := u.running
	u.mu.Unlock()
	if running {
		return false
	}

	lastUpload := u.readMarker()
	entries, err := os.ReadDir(u.dailyDir)
	if err != nil {
		return false
	}

	sectionRe := regexp.MustCompile(`^## (\d{2}:\d{2}:\d{2})`)
	count := 0

	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".md") {
			continue
		}
		stem := strings.TrimSuffix(e.Name(), ".md")
		fileDate, err := time.Parse("2006-01-02", stem)
		if err != nil {
			continue
		}
		if lastUpload != nil && fileDate.Before(lastUpload.Truncate(24*time.Hour)) {
			continue
		}

		content, err := os.ReadFile(filepath.Join(u.dailyDir, e.Name()))
		if err != nil {
			continue
		}

		for _, line := range strings.Split(string(content), "\n") {
			m := sectionRe.FindStringSubmatch(line)
			if m == nil {
				continue
			}
			if lastUpload != nil {
				t, err := time.Parse("15:04:05", m[1])
				if err != nil {
					count++
					continue
				}
				sectionDT := time.Date(fileDate.Year(), fileDate.Month(), fileDate.Day(),
					t.Hour(), t.Minute(), t.Second(), 0, time.Local)
				if sectionDT.After(*lastUpload) {
					count++
				}
			} else {
				count++
			}
		}

		if count >= uploadChunkThreshold {
			return true
		}
	}
	return count >= uploadChunkThreshold
}

// UploadOnce creates a tar.gz snapshot of the memory directory.
func (u *Uploader) UploadOnce(ctx context.Context) error {
	u.mu.Lock()
	if u.running {
		u.mu.Unlock()
		return nil
	}
	u.running = true
	u.mu.Unlock()

	defer func() {
		u.mu.Lock()
		u.running = false
		u.mu.Unlock()
	}()

	tmpFile, err := os.CreateTemp("", "memory-upload-*.tar.gz")
	if err != nil {
		return fmt.Errorf("uploader: create temp file: %w", err)
	}
	tmpPath := tmpFile.Name()
	defer os.Remove(tmpPath)

	if err := u.createTarGz(tmpFile); err != nil {
		_ = tmpFile.Close()
		return fmt.Errorf("uploader: create archive: %w", err)
	}
	_ = tmpFile.Close()

	if err := u.upload(ctx, tmpPath); err != nil {
		return fmt.Errorf("uploader: upload: %w", err)
	}

	u.writeMarker()
	u.log.Info("memory upload complete")
	return nil
}

// createTarGz generate a tar.gz of memory.
func (u *Uploader) createTarGz(w io.Writer) error {
	gzw := gzip.NewWriter(w)
	defer func() { _ = gzw.Close() }()
	tw := tar.NewWriter(gzw)
	defer func() { _ = tw.Close() }()

	return filepath.Walk(u.memoryRoot, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return nil // skip unreadable files
		}

		relPath, err := filepath.Rel(u.memoryRoot, path)
		if err != nil {
			return nil
		}

		if relPath == "index" || strings.HasPrefix(relPath, "index/") {
			if info.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}

		if strings.HasSuffix(relPath, ".tar.gz") {
			return nil
		}

		header, err := tar.FileInfoHeader(info, "")
		if err != nil {
			return nil
		}
		header.Name = filepath.Join("memory", relPath)

		if err := tw.WriteHeader(header); err != nil {
			return err
		}

		if info.IsDir() {
			return nil
		}

		f, err := os.Open(path)
		if err != nil {
			return nil
		}
		defer func() { _ = f.Close() }()

		_, err = io.Copy(tw, f)
		return err
	})
}

// upload sends the tar.gz file as a multipart PUT request.
func (u *Uploader) upload(ctx context.Context, tarPath string) error {
	ctx, cancel := context.WithTimeout(ctx, uploadTimeout)
	defer cancel()

	f, err := os.Open(tarPath)
	if err != nil {
		return err
	}
	defer func() { _ = f.Close() }()

	pr, pw := io.Pipe()
	writer := multipart.NewWriter(pw)

	go func() {
		part, err := writer.CreateFormFile("memory", "memory.tar.gz")
		if err != nil {
			_ = pw.CloseWithError(err)
			return
		}
		if _, err := io.Copy(part, f); err != nil {
			_ = pw.CloseWithError(err)
			return
		}
		_ = writer.Close()
		_ = pw.Close()
	}()

	req, err := http.NewRequestWithContext(ctx, http.MethodPut, u.uploadURL, pr)
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", writer.FormDataContentType())
	req.Header.Set("Authorization", "Bearer "+u.apiKey)

	resp, err := u.client.Do(req)
	if err != nil {
		return fmt.Errorf("upload request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return fmt.Errorf("upload returned %s: %s", resp.Status, body)
	}

	return nil
}

func (u *Uploader) readMarker() *time.Time {
	raw, err := os.ReadFile(u.markerFile)
	if err != nil {
		return nil
	}
	t, err := time.ParseInLocation("2006-01-02 15:04", strings.TrimSpace(string(raw)), time.Local)
	if err != nil {
		return nil
	}
	return &t
}

func (u *Uploader) writeMarker() {
	_ = os.WriteFile(u.markerFile, []byte(time.Now().Format("2006-01-02 15:04")), 0o644)
}
