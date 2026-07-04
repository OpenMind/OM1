package memory

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/httpclient"
	"github.com/openmind/om1/internal/knowledgebase"
)

// Uploader handles all cloud uploads: daily logs, profiles, summarize trigger.
type Uploader struct {
	apiURL   string
	apiKey   string
	embedder knowledgebase.Embedder
	client   *http.Client
	log      *zap.Logger
}

// NewUploader creates an Uploader.
func NewUploader(apiURL, apiKey, embedderURL string, log *zap.Logger) *Uploader {
	return &Uploader{
		apiURL:   apiURL,
		apiKey:   apiKey,
		embedder: knowledgebase.NewHTTPEmbedder(embedderURL),
		client:   httpclient.Default(),
		log:      log,
	}
}

// PostDailyLog embeds the content locally and uploads to cloud with the vector.
func (u *Uploader) PostDailyLog(ctx context.Context, content, faceUUID string) {
	if strings.TrimSpace(content) == "" {
		return
	}

	embedding, err := u.embedder.Embed(ctx, content)
	if err != nil {
		u.log.Warn("embed for upload failed", zap.Error(err))
		embedding = nil // upload without embedding
	}

	type entry struct {
		FaceUUID  string    `json:"face_uuid,omitempty"`
		Timestamp string    `json:"timestamp"`
		Content   string    `json:"content"`
		Embedding []float32 `json:"embedding,omitempty"`
	}

	body, _ := json.Marshal(map[string][]entry{
		"entries": {{
			FaceUUID:  faceUUID,
			Timestamp: time.Now().UTC().Format(time.RFC3339),
			Content:   content,
			Embedding: embedding,
		}},
	})

	u.doPost(ctx, "/memory/daily-logs", body, "daily log")
}

// PostUserProfile uploads a profile update to the cloud.
func (u *Uploader) PostUserProfile(ctx context.Context, faceUUID string, names []string, interactionCount, visitCount int) {
	if faceUUID == "" {
		return
	}

	now := time.Now().UTC().Format(time.RFC3339)
	type profile struct {
		FaceUUID         string   `json:"face_uuid"`
		Names            []string `json:"names"`
		FirstSeen        string   `json:"first_seen"`
		LastSeen         string   `json:"last_seen"`
		InteractionCount int      `json:"interaction_count"`
		VisitCount       int      `json:"visit_count"`
	}

	body, _ := json.Marshal(map[string][]profile{
		"profiles": {{
			FaceUUID:         faceUUID,
			Names:            names,
			FirstSeen:        now,
			LastSeen:         now,
			InteractionCount: interactionCount,
			VisitCount:       visitCount,
		}},
	})

	u.doPost(ctx, "/memory/users", body, "user profile")
}

// TriggerSummarize triggers async summarization on the backend.
func (u *Uploader) TriggerSummarize(ctx context.Context) {
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, u.apiURL+"/memory/summarize", nil)
	if err != nil {
		u.log.Warn("summarize trigger request build failed", zap.Error(err))
		return
	}
	req.Header.Set("Authorization", "Bearer "+u.apiKey)

	resp, err := u.client.Do(req)
	if err != nil {
		u.log.Warn("summarize trigger failed", zap.Error(err))
		return
	}
	defer func() { _ = resp.Body.Close() }()
	_, _ = io.ReadAll(resp.Body)

	if resp.StatusCode == http.StatusOK {
		u.log.Debug("summarize triggered")
	}
}

// PostSignals uploads recall signals from search results to the cloud.
func (u *Uploader) PostSignals(ctx context.Context, chunks []MemoryEntry, userID, queryHash string) {
	if len(chunks) == 0 {
		return
	}

	type entry struct {
		ChunkHash    string   `json:"chunk_hash"`
		RecallCount  int      `json:"recall_count"`
		TotalScore   float64  `json:"total_score"`
		QueryHashes  []string `json:"query_hashes"`
		RecallDays   []string `json:"recall_days"`
		LastRecalled string   `json:"last_recalled"`
		FirstSeen    string   `json:"first_seen"`
		Text         *string  `json:"text,omitempty"`
		UserID       *string  `json:"user_id,omitempty"`
	}

	now := time.Now().UTC()
	nowStr := now.Format(time.RFC3339)
	dayStr := now.Format("2006-01-02")

	entries := make([]entry, 0, len(chunks))
	for _, c := range chunks {
		hash := fmt.Sprintf("%x", sha256.Sum256([]byte(c.Text)))[:32]
		e := entry{
			ChunkHash:    hash,
			RecallCount:  1,
			TotalScore:   c.Score,
			QueryHashes:  []string{queryHash},
			RecallDays:   []string{dayStr},
			LastRecalled: nowStr,
			FirstSeen:    nowStr,
			Text:         &c.Text,
		}
		if userID != "" {
			e.UserID = &userID
		}
		entries = append(entries, e)
	}

	body, _ := json.Marshal(map[string][]entry{"signals": entries})
	u.doPost(ctx, "/memory/signals", body, "signals")
}

func (u *Uploader) doPost(ctx context.Context, path string, body []byte, label string) {
	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, u.apiURL+path, bytes.NewReader(body))
	if err != nil {
		u.log.Warn("upload request build failed", zap.String("label", label), zap.Error(err))
		return
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+u.apiKey)

	resp, err := u.client.Do(req)
	if err != nil {
		u.log.Warn("upload failed", zap.String("label", label), zap.Error(err))
		return
	}
	defer func() { _ = resp.Body.Close() }()
	_, _ = io.ReadAll(resp.Body)

	if resp.StatusCode != http.StatusOK {
		u.log.Warn("upload returned non-200",
			zap.String("label", label),
			zap.Int("status", resp.StatusCode),
		)
	}
}
