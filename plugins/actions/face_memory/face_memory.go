package face_memory

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/actions"
	"github.com/openmind/om1/internal/httpclient"
	"github.com/openmind/om1/internal/logger"
	"github.com/openmind/om1/internal/providers"
	"github.com/openmind/om1/internal/providers/tts"
	"github.com/openmind/om1/internal/util"
)

const (
	faceMemoryDefaultBaseURL        = "http://127.0.0.1:6793"
	faceMemoryDefaultRecentSec      = 1.0
	faceMemoryDefaultPollMs         = 200
	faceMemoryDefaultTimeoutSec     = 8
	faceMemoryDefaultHTTPTimeoutSec = 5.0
	faceMemoryDefaultTopK           = 3
	faceMemoryDefaultMinSim         = 0.30
)

// FaceMemoryOp is the enum of supported operations.
type FaceMemoryOp string

func (FaceMemoryOp) EnumValues() []string {
	return []string{
		"selfie",
		"correct_identity",
		"forget_last",
		"find_similar",
		"merge",
		"set_name",
	}
}

// FaceMemoryInput is the LLM-facing schema for the unified action.
type FaceMemoryInput struct {
	Op          FaceMemoryOp `json:"op"                     description:"Which face memory operation to perform."`
	ID          string       `json:"id,omitempty"           description:"Name for op=selfie ONLY. For op=set_name use to_id instead."`
	Force       bool         `json:"force,omitempty"        description:"Bypass cross-name reject for selfie (default false). Only used when op=selfie."`
	FromID      string       `json:"from_id,omitempty"      description:"Current (wrong) id for correct_identity."`
	ToID        string       `json:"to_id,omitempty"        description:"The name to store. REQUIRED for op=set_name and op=correct_identity."`
	TargetName  string       `json:"target_name,omitempty"  description:"Confirmed name for merge (e.g. 'sean')."`
	ConfirmedBy string       `json:"confirmed_by,omitempty" description:"How merge was confirmed (e.g. 'user_voice')."`
	TopK        int          `json:"top_k,omitempty"        description:"Candidates for find_similar (default 3)."`
	MinSim      float64      `json:"min_sim,omitempty"      description:"Min cosine sim for find_similar (default 0.30)."`
}

func init() {
	actions.RegisterInterface(
		"face_memory",
		"Unified face memory management. Use the 'op' field to select the operation: "+
			"selfie (enroll a person), correct_identity (fix a name typo), "+
			"forget_last (undo last enrollment), find_similar (search gallery), "+
			"merge (combine anonymous face into named identity), "+
			"set_name (rename person currently visible).",
		FaceMemoryInput{},
	)
	actions.Register("face_memory/face_memory", NewConnector)
}

type Config struct {
	APIKey           string  `json:"api_key"`
	ElevenLabsAPIKey string  `json:"elevenlabs_api_key"`
	VoiceID          string  `json:"voice_id"`
	ModelID          string  `json:"model_id"`
	OutputFormat     string  `json:"output_format"`
	Rate             int     `json:"rate"`
	FaceHTTPBaseURL  string  `json:"face_http_base_url"`
	FaceRecentSec    float64 `json:"face_recent_sec"`
	PollMs           int     `json:"poll_ms"`
	TimeoutSec       int     `json:"timeout_sec"`
	HTTPTimeoutSec   float64 `json:"http_timeout_sec"`
	DefaultTopK      int     `json:"default_top_k"`
	DefaultMinSim    float64 `json:"default_min_sim"`
}

type Connector struct {
	log    *zap.Logger
	cfg    Config
	client *http.Client
	tts    *tts.ElevenLabsProvider

	mu             sync.Mutex
	lastEnrolledID string
	lastMatchName  string
}

// var dedupSuffixRE = regexp.MustCompile(`_\d+$`)

func NewConnector(configMap map[string]any) (actions.Connector, error) {
	var cfg Config
	if b, err := json.Marshal(configMap); err == nil {
		_ = json.Unmarshal(b, &cfg)
	}
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("face_memory: api_key required")
	}
	if cfg.FaceHTTPBaseURL == "" {
		cfg.FaceHTTPBaseURL = faceMemoryDefaultBaseURL
	}
	if cfg.FaceRecentSec == 0 {
		cfg.FaceRecentSec = faceMemoryDefaultRecentSec
	}
	if cfg.PollMs == 0 {
		cfg.PollMs = faceMemoryDefaultPollMs
	}
	if cfg.TimeoutSec == 0 {
		cfg.TimeoutSec = faceMemoryDefaultTimeoutSec
	}
	if cfg.HTTPTimeoutSec == 0 {
		cfg.HTTPTimeoutSec = faceMemoryDefaultHTTPTimeoutSec
	}
	if cfg.VoiceID == "" {
		cfg.VoiceID = tts.DefaultVoiceID
	}
	if cfg.ModelID == "" {
		cfg.ModelID = tts.DefaultModelID
	}
	if cfg.OutputFormat == "" {
		cfg.OutputFormat = tts.DefaultOutputFormat
	}
	if cfg.Rate == 0 {
		cfg.Rate = tts.DefaultRate
	}
	if cfg.DefaultTopK == 0 {
		cfg.DefaultTopK = faceMemoryDefaultTopK
	}
	if cfg.DefaultMinSim == 0 {
		cfg.DefaultMinSim = faceMemoryDefaultMinSim
	}

	log := logger.Get()
	ttsClient := tts.ElevenLabs(tts.ElevenLabsConfig{
		APIKey:           cfg.APIKey,
		ElevenLabsAPIKey: cfg.ElevenLabsAPIKey,
		VoiceID:          cfg.VoiceID,
		ModelID:          cfg.ModelID,
		OutputFormat:     cfg.OutputFormat,
		Rate:             cfg.Rate,
	}, log)

	return &Connector{
		log: log,
		cfg: cfg,
		client: &http.Client{
			Transport: httpclient.Default().Transport,
			Timeout:   time.Duration(cfg.HTTPTimeoutSec * float64(time.Second)),
		},
		tts: ttsClient,
	}, nil
}

func (c *Connector) Connect(ctx context.Context, input actions.Input) (actions.Output, error) {
	args, ok := input.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("face_memory: unexpected input type %T", input)
	}
	op := util.StringFrom(args["op"], "")
	switch op {
	case "selfie":
		return c.doSelfie(ctx, args)
	case "correct_identity":
		return c.doCorrectIdentity(ctx, args)
	case "forget_last":
		return c.doForgetLast(ctx, args)
	case "find_similar":
		return c.doFindSimilar(ctx, args)
	case "merge":
		return c.doMerge(ctx, args)
	case "set_name":
		return c.doSetName(ctx, args)
	default:
		c.writeStatus(fmt.Sprintf("result=unknown_op op=%s", op))
		return nil, nil
	}
}

func (c *Connector) Tick(ctx context.Context) { <-ctx.Done() }
func (c *Connector) Stop()                    {}

func (c *Connector) doSelfie(ctx context.Context, args map[string]any) (any, error) {
	if spk := providers.Speaker().Latest(); spk.Identified() && spk.UUID != "" {
		c.log.Info("face_memory/selfie: speaker already enrolled, naming instead",
			zap.String("uuid", shortUUID(spk.UUID)), zap.Int("track", spk.TrackID))
		return c.doSetName(ctx, map[string]any{
			"to_id": util.StringFrom(args["id"], ""),
		})
	}

	name := strings.TrimSpace(util.StringFrom(args["id"], ""))
	if name == "" {
		c.writeStatus("result=bad_id detail=empty")
		c.log.Error("face_memory/selfie: empty id")
		return nil, nil
	}

	timeoutSec := util.IntFrom(args["timeout_sec"], c.cfg.TimeoutSec)
	force := util.BoolFrom(args["force"], false)

	// Disable blur for enrollment so the demo view shows the face.
	origBlur := c.getBlur()
	c.setBlur(false)
	defer c.setBlur(origBlur)

	if !c.waitAnyFace(ctx, timeoutSec) {
		c.writeStatus("result=low_quality reason=no_one_present")
		// c.speak("I don't see anyone in front of me yet.")
		return nil, nil
	}

	body := map[string]any{"id": name, "force": force}
	resp := c.postJSON("/selfie", body)

	// Retry once on transient busy.
	if resp != nil {
		if errStr, _ := resp["error"].(string); errStr == "busy" {
			c.log.Info("face_memory/selfie: busy, retrying in 1s")
			select {
			case <-time.After(1 * time.Second):
			case <-ctx.Done():
				return nil, nil
			}
			resp = c.postJSON("/selfie", body)
		}
	}

	if resp == nil {
		c.dispatchNetworkError()
	} else {
		c.dispatchSelfieResponse(resp, name)
	}
	return nil, nil
}

func (c *Connector) dispatchSelfieResponse(resp map[string]any, claimedID string) {
	if ok, _ := resp["ok"].(bool); ok {
		savedID := util.StringFrom(resp["id"], claimedID)
		merged := util.BoolFrom(resp["merged"], false)
		samples := util.IntFrom(resp["samples_saved"], 0)
		// display := displayName(savedID)

		c.mu.Lock()
		c.lastEnrolledID = savedID
		c.lastMatchName = ""
		c.mu.Unlock()

		tag := "success"
		if merged {
			tag = "merged"
		}
		c.writeStatus(fmt.Sprintf("result=%s id=%s samples=%d merged=%t",
			tag, savedID, samples, merged))
		// if merged {
		// 	c.speak(fmt.Sprintf("Welcome back, %s!", display))
		// } else {
		// 	c.speak(fmt.Sprintf("Nice to meet you, %s! I'll remember you next time.", display))
		// }
		c.log.Info("face_memory/selfie: enroll ok",
			zap.String("tag", tag), zap.String("id", savedID), zap.Int("samples", samples))
		return
	}

	errStr := util.StringFrom(resp["error"], "unknown")
	switch errStr {
	case "ambiguous_subjects":
		n := util.IntFrom(resp["n_engaged"], 0)
		c.writeStatus(fmt.Sprintf("result=ambiguous engaged=%d", n))
		// c.speak("I see a few people. Could you step closer so I can focus on you?")
		c.clearState()
	case "face_belongs_to":
		matched := util.StringFrom(resp["name"], "someone")
		sim := util.FloatFrom(resp["sim"], 0.0)
		// display := displayName(matched)
		c.mu.Lock()
		c.lastMatchName = matched
		c.mu.Unlock()
		c.writeStatus(fmt.Sprintf("result=face_belongs_to claimed=%s matched=%s sim=%.3f",
			claimedID, matched, sim))
		// c.speak(fmt.Sprintf("You look a lot like %s. Are you %s, or someone different?",
		// display, display))
	case "no_valid_frames":
		c.writeStatus("result=low_quality")
		// c.speak("I can't see your face clearly. Could you look at me directly?")
		c.clearState()
	case "insufficient_samples":
		got := util.IntFrom(resp["got"], 0)
		c.writeStatus(fmt.Sprintf("result=partial got=%d", got))
		// c.speak("Hold still — almost got it.")
		c.clearState()
	case "busy":
		c.writeStatus("result=busy retries=1")
		// c.speak("One sec, finishing the last one.")
		c.clearState()
	case "bad_id":
		detail := util.StringFrom(resp["detail"], "")
		c.writeStatus(fmt.Sprintf("result=bad_id detail=%s", detail))
		c.clearState()
	case "recognition_disabled":
		c.writeStatus("result=recognition_disabled")
		// c.speak("I can't see right now — please try again in a moment.")
		c.clearState()
	default:
		c.writeStatus(fmt.Sprintf("result=unknown error=%s", errStr))
		// c.speak("Something went wrong. Could you try again?")
		c.clearState()
	}
}

func (c *Connector) doCorrectIdentity(_ context.Context, args map[string]any) (any, error) {
	fromID := normID(args, "from_id")
	toID := normID(args, "to_id")

	if fromID == "" || toID == "" {
		c.writeStatus(fmt.Sprintf("result=bad_id from=%q to=%q", fromID, toID))
		c.log.Error("face_memory/correct_identity: missing ids",
			zap.String("from_id", fromID), zap.String("to_id", toID))
		return nil, nil
	}
	if fromID == toID {
		c.writeStatus(fmt.Sprintf("result=same_id id=%s", fromID))
		return nil, nil
	}

	uuid, count, err := c.resolveNameToUUID(fromID)
	if err != nil {
		c.writeStatus("result=network_error")
		// c.speak("I had trouble updating that.")
		return nil, nil
	}
	switch count {
	case 0:
		c.writeStatus(fmt.Sprintf("result=name_not_found from=%s", fromID))
		// c.speak(fmt.Sprintf("I don't have anyone named %s saved.", displayName(fromID)))
		return nil, nil
	case 1:
		// fall through
	default:
		c.writeStatus(fmt.Sprintf("result=ambiguous from=%s count=%d", fromID, count))
		// c.speak(fmt.Sprintf("I have more than one %s saved — I can't tell which one you meant.", displayName(fromID)))
		return nil, nil
	}

	resp := c.postJSON("/set_name", map[string]any{"uuid": uuid, "name": toID})
	c.dispatchRenameResponse(resp, fromID, toID, uuid)
	return nil, nil
}

func (c *Connector) resolveNameToUUID(name string) (string, int, error) {
	resp := c.postJSON("/gallery/identities", map[string]any{})
	if resp == nil {
		return "", 0, fmt.Errorf("network error")
	}
	rawList, _ := resp["identities"].([]any)
	target := strings.ToLower(strings.TrimSpace(name))
	var matches []string
	for _, item := range rawList {
		m, ok := item.(map[string]any)
		if !ok {
			continue
		}
		n, _ := m["name"].(string)
		if strings.ToLower(strings.TrimSpace(n)) == target {
			u, _ := m["uuid"].(string)
			if u != "" {
				matches = append(matches, u)
			}
		}
	}
	if len(matches) == 0 {
		return "", 0, nil
	}
	if len(matches) > 1 {
		return "", len(matches), nil
	}
	return matches[0], 1, nil
}

func (c *Connector) dispatchRenameResponse(resp map[string]any, fromID, toID, uuid string) {
	if resp == nil {
		c.writeStatus("result=network_error")
		// c.speak("I had trouble updating that.")
		return
	}
	if ok, _ := resp["ok"].(bool); ok {
		samples := util.IntFrom(resp["sample_count"], 0)
		c.writeStatus(fmt.Sprintf(
			"result=success from=%s to=%s uuid=%s samples=%d",
			fromID, toID, shortUUID(uuid), samples,
		))
		// c.speak(fmt.Sprintf("Got it, I've updated your name to %s.", displayName(toID)))
		c.log.Info("face_memory/correct_identity: ok",
			zap.String("from", fromID), zap.String("to", toID),
			zap.String("uuid", uuid), zap.Int("samples", samples))
		return
	}
	errStr := util.StringFrom(resp["error"], "unknown")
	switch errStr {
	case "uuid_not_found":
		c.writeStatus(fmt.Sprintf("result=uuid_not_found uuid=%s", shortUUID(uuid)))
		// c.speak("I couldn't find that identity anymore — maybe it was deleted.")
	case "bad_name":
		// detail := util.StringFrom(resp["detail"], "")
		// c.writeStatus(fmt.Sprintf("result=bad_id detail=%s", detail))
	case "recognition_disabled":
		c.writeStatus("result=recognition_disabled")
		// c.speak("I can't update names right now.")
	default:
		c.writeStatus(fmt.Sprintf("result=unknown error=%s", errStr))
		// c.speak("Something went wrong updating that.")
	}
}

func (c *Connector) doForgetLast(_ context.Context, _ map[string]any) (any, error) {
	resp := c.postJSON("/gallery/forget_last", map[string]any{})
	c.dispatchForgetResponse(resp)
	return nil, nil
}

func (c *Connector) dispatchForgetResponse(resp map[string]any) {
	if resp == nil {
		c.writeStatus("result=network_error")
		// c.speak("I couldn't undo that.")
		return
	}
	if ok, _ := resp["ok"].(bool); ok {
		uuid := util.StringFrom(resp["uuid"], "")
		name := util.StringFrom(resp["name"], "")
		c.writeStatus(fmt.Sprintf("result=success uuid=%s name=%s", shortUUID(uuid), name))
		// if name != "" {
		// 	c.speak(fmt.Sprintf("OK, I've forgotten %s. Let's try again.", displayName(name)))
		// } else {
		// 	c.speak("OK, I've forgotten that one. Let's try again.")
		// }
		c.log.Info("face_memory/forget_last: ok",
			zap.String("uuid", uuid), zap.String("name", name))
		return
	}
	errStr := util.StringFrom(resp["error"], "unknown")
	switch errStr {
	case "no_recent_enrollment":
		c.writeStatus("result=no_recent_enrollment")
		// c.speak("There's nothing recent for me to forget.")
	case "stale_enrollment":
		c.writeStatus("result=stale_enrollment")
		// c.speak("Too much time has passed — I can't undo that anymore.")
	case "uuid_mismatch":
		detail := util.StringFrom(resp["detail"], "")
		c.writeStatus(fmt.Sprintf("result=uuid_mismatch detail=%s", detail))
	case "recognition_disabled":
		c.writeStatus("result=recognition_disabled")
		// c.speak("I can't undo that right now.")
	default:
		c.writeStatus(fmt.Sprintf("result=unknown error=%s", errStr))
		// c.speak("Something went wrong undoing that.")
	}
}

func (c *Connector) doFindSimilar(_ context.Context, args map[string]any) (any, error) {
	topK := util.IntFrom(args["top_k"], c.cfg.DefaultTopK)
	if topK < 1 {
		topK = c.cfg.DefaultTopK
	}
	minSim := util.FloatFrom(args["min_sim"], c.cfg.DefaultMinSim)

	resp := c.postJSON("/gallery/find_similar_current", map[string]any{
		"top_k":   topK,
		"min_sim": minSim,
	})
	c.dispatchFindSimilarResponse(resp)
	return nil, nil
}

func (c *Connector) dispatchFindSimilarResponse(resp map[string]any) {
	if resp == nil {
		c.writeSimilarMatches("result=network_error")
		return
	}
	if ok, _ := resp["ok"].(bool); ok {
		rawMatches, _ := resp["matches"].([]any)
		if len(rawMatches) == 0 {
			c.writeSimilarMatches("result=no_match matches=0")
			return
		}
		parts := []string{
			"result=success",
			fmt.Sprintf("matches=%d", len(rawMatches)),
		}
		for i, m := range rawMatches {
			match, ok := m.(map[string]any)
			if !ok {
				continue
			}
			name := util.StringFrom(match["name"], "")
			sim := util.FloatFrom(match["sim"], 0.0)
			label := name
			if label == "" {
				uuid := util.StringFrom(match["uuid"], "")
				if len(uuid) >= 6 {
					label = "anon_" + uuid[:6]
				} else {
					label = "anon"
				}
			}
			slot := "extra"
			switch i {
			case 0:
				slot = "best"
			case 1:
				slot = "second"
			case 2:
				slot = "third"
			}
			parts = append(parts, fmt.Sprintf("%s=%s(%.2f)", slot, label, sim))
		}
		c.writeSimilarMatches(strings.Join(parts, " "))
		c.log.Info("face_memory/find_similar: ok", zap.Int("matches", len(rawMatches)))
		return
	}
	errStr := util.StringFrom(resp["error"], "unknown")
	switch errStr {
	case "no_visible_face":
		c.writeSimilarMatches("result=no_visible_face")
	case "uuid_not_found":
		c.writeSimilarMatches("result=uuid_not_found")
	case "no_centroid":
		c.writeSimilarMatches("result=no_centroid")
	case "bad_top_k", "bad_min_sim":
		c.writeSimilarMatches("result=" + errStr)
	case "recognition_disabled":
		c.writeSimilarMatches("result=recognition_disabled")
	default:
		c.writeSimilarMatches(fmt.Sprintf("result=unknown error=%s", errStr))
	}
}

func (c *Connector) doMerge(_ context.Context, args map[string]any) (any, error) {
	targetName := strings.ToLower(strings.TrimSpace(util.StringFrom(args["target_name"], "")))
	confirmedBy := strings.TrimSpace(util.StringFrom(args["confirmed_by"], ""))

	if targetName == "" {
		c.writeStatus("result=missing_target_name")
		c.log.Warn("face_memory/merge: missing target_name")
		return nil, nil
	}
	if confirmedBy == "" {
		confirmedBy = "llm_unattributed"
	}

	resp := c.postJSON("/gallery/merge_current", map[string]any{
		"target_name":  targetName,
		"confirmed_by": confirmedBy,
	})
	c.dispatchMergeResponse(resp, targetName)
	return nil, nil
}

func (c *Connector) dispatchMergeResponse(resp map[string]any, targetName string) {
	if resp == nil {
		c.writeStatus("result=network_error")
		// c.speak("I had trouble combining those identities.")
		return
	}
	if ok, _ := resp["ok"].(bool); ok {
		samples := util.IntFrom(resp["samples_merged"], 0)
		sim := util.FloatFrom(resp["sim"], 0.0)
		c.writeStatus(fmt.Sprintf(
			"result=success target_name=%s samples=%d sim=%.2f",
			targetName, samples, sim,
		))
		c.log.Info("face_memory/merge: ok",
			zap.String("target_name", targetName),
			zap.Int("samples", samples), zap.Float64("sim", sim))
		return
	}
	errStr := util.StringFrom(resp["error"], "unknown")
	switch errStr {
	case "missing_target_name":
		c.writeStatus("result=missing_target_name")
	case "no_visible_face":
		c.writeStatus("result=no_visible_face")
		// c.speak("I can't see anyone in front of me right now.")
	case "source_is_named":
		srcName := util.StringFrom(resp["source_name"], "")
		c.writeStatus(fmt.Sprintf("result=source_is_named source_name=%s", srcName))
	case "target_name_not_found":
		c.writeStatus(fmt.Sprintf("result=target_name_not_found target_name=%s", targetName))
	case "ambiguous_target":
		c.writeStatus(fmt.Sprintf("result=ambiguous_target target_name=%s", targetName))
		// c.speak("I have a few people with that name — could you help me figure out which?")
	case "no_centroid":
		c.writeStatus("result=no_centroid")
	case "same_uuid":
		c.writeStatus("result=same_uuid")
	case "uuid_not_found":
		role := util.StringFrom(resp["role"], "?")
		c.writeStatus(fmt.Sprintf("result=uuid_not_found role=%s", role))
	case "recognition_disabled":
		c.writeStatus("result=recognition_disabled")
	default:
		c.writeStatus(fmt.Sprintf("result=unknown error=%s", errStr))
		// c.speak("Something went wrong combining those identities.")
	}
}

func (c *Connector) doSetName(_ context.Context, args map[string]any) (any, error) {
	toID := normID(args, "to_id")
	if toID == "" {
		toID = normID(args, "id")
	}
	if toID == "" {
		c.writeStatus("result=bad_id to=")
		c.log.Error("face_memory/set_name: missing to_id")
		return nil, nil
	}

	if spk := providers.Speaker().Latest(); spk.Identified() {
		body := map[string]any{"name": toID, "confirmed_by": "user_voice"}
		if spk.UUID != "" {
			body["uuid"] = spk.UUID
		} else {
			body["track_id"] = spk.TrackID
		}

		resp := c.postJSON("/set_name", body)

		if spk.UUID != "" && resp != nil {
			if errStr, _ := resp["error"].(string); errStr == "uuid_not_found" {
				c.log.Info("face_memory/set_name: uuid gone, retrying by track",
					zap.String("uuid", shortUUID(spk.UUID)), zap.Int("track", spk.TrackID))
				resp = c.postJSON("/set_name", map[string]any{
					"name":         toID,
					"track_id":     spk.TrackID,
					"confirmed_by": "user_voice",
				})
			}
		}

		c.dispatchSetNameCurrentResponse(resp, toID)
		return nil, nil
	}

	dominant, n := dominantFace()
	switch {
	case n == 0 || n == 1:
		c.log.Info("face_memory/set_name: single face, no ambiguity",
			zap.String("name", toID))
		resp := c.postJSON("/set_name_current", map[string]any{
			"name":         toID,
			"confirmed_by": "user_voice",
		})
		c.dispatchSetNameCurrentResponse(resp, toID)
		return nil, nil

	case dominant != nil:
		c.log.Info("face_memory/set_name: no measured speaker, one face clearly dominant",
			zap.Int("track", dominant.TrackID), zap.Int("visible_faces", n),
			zap.String("name", toID))
		body := map[string]any{"name": toID, "confirmed_by": "user_voice"}
		if dominant.UUID != nil && *dominant.UUID != "" {
			body["uuid"] = *dominant.UUID
		} else {
			body["track_id"] = dominant.TrackID
		}
		resp := c.postJSON("/set_name", body)
		c.dispatchSetNameCurrentResponse(resp, toID)
		return nil, nil

	default:
		c.writeStatus(fmt.Sprintf("result=ambiguous_speaker faces=%d name=%s", n, toID))
		c.log.Info("face_memory/set_name: declined, several faces and none dominant",
			zap.Int("visible_faces", n), zap.String("name", toID))
		return nil, nil
	}
}

const faceDominanceRatio = 2.5

func dominantFace() (*providers.SpeakerFace, int) {
	spk := providers.Speaker().Latest()
	if spk == nil || len(spk.Faces) == 0 {
		return nil, 0
	}
	faces := spk.Faces
	if len(faces) == 1 {
		return &faces[0], 1
	}

	first, second := -1, -1
	for i := range faces {
		if first < 0 || faces[i].Area > faces[first].Area {
			second, first = first, i
		} else if second < 0 || faces[i].Area > faces[second].Area {
			second = i
		}
	}
	if first < 0 || second < 0 || faces[second].Area <= 0 {
		return nil, len(faces)
	}
	ratio := float64(faces[first].Area) / float64(faces[second].Area)
	if ratio < faceDominanceRatio {
		return nil, len(faces)
	}
	return &faces[first], len(faces)
}

func (c *Connector) dispatchSetNameCurrentResponse(resp map[string]any, toID string) {
	if resp == nil {
		c.writeStatus("result=network_error")
		// c.speak("I had trouble updating that name.")
		return
	}
	if ok, _ := resp["ok"].(bool); ok {
		name := util.StringFrom(resp["name"], toID)
		prev := util.StringFrom(resp["prev_name"], "")
		uuid := util.StringFrom(resp["uuid"], "")
		c.writeStatus(fmt.Sprintf(
			"result=success name=%s prev_name=%s uuid=%s",
			name, prev, shortUUID(uuid),
		))
		// c.speak(fmt.Sprintf("Got it — I'll call you %s from now on.", displayName(name)))
		c.log.Info("face_memory/set_name: ok",
			zap.String("name", name), zap.String("prev", prev), zap.String("uuid", uuid))
		return
	}
	errStr := util.StringFrom(resp["error"], "unknown")
	switch errStr {
	case "no_visible_face":
		c.writeStatus("result=no_visible_face")
		// c.speak("I can't see anyone in front of me right now.")
	case "bad_name":
		detail := util.StringFrom(resp["detail"], "")
		c.writeStatus(fmt.Sprintf("result=bad_id detail=%s", detail))
	case "uuid_not_found":
		c.writeStatus("result=uuid_not_found")
		// c.speak("I couldn't find that person anymore.")
	case "track_not_identified":
		detail := util.StringFrom(resp["detail"], "")
		c.writeStatus(fmt.Sprintf("result=track_not_identified detail=%s", detail))
	case "recognition_disabled":
		c.writeStatus("result=recognition_disabled")
		// c.speak("I can't update names right now.")
	default:
		c.writeStatus(fmt.Sprintf("result=unknown error=%s", errStr))
		// c.speak("Something went wrong updating that name.")
	}
}

func (c *Connector) postJSON(path string, body map[string]any) map[string]any {
	url := c.cfg.FaceHTTPBaseURL + path
	buf, _ := json.Marshal(body)
	req, err := http.NewRequest("POST", url, bytes.NewReader(buf))
	if err != nil {
		c.log.Warn("face_memory: build request failed", zap.String("url", url), zap.Error(err))
		return nil
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.client.Do(req)
	if err != nil {
		c.log.Warn("face_memory: HTTP error", zap.String("url", url), zap.Error(err))
		return nil
	}
	defer func() { _ = resp.Body.Close() }()
	data, _ := io.ReadAll(resp.Body)
	var out map[string]any
	if err := json.Unmarshal(data, &out); err != nil {
		c.log.Warn("face_memory: decode failed", zap.String("url", url), zap.Error(err))
		return nil
	}
	return out
}

func (c *Connector) writeStatus(line string) {
	providers.IO().AddInput("SelfieStatus", line, time.Time{})
	c.log.Info("face_memory: status", zap.String("line", line))
}

func (c *Connector) writeSimilarMatches(line string) {
	providers.IO().AddInput("SimilarMatches", line, time.Time{})
	c.log.Info("face_memory: similar_matches", zap.String("line", line))
}

// func (c *Connector) speak(message string) {
// 	if message == "" {
// 		return
// 	}
// 	c.tts.AddText(message)
// }

func (c *Connector) clearState() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.lastEnrolledID = ""
	c.lastMatchName = ""
}

func (c *Connector) dispatchNetworkError() {
	c.writeStatus("result=network_error")
	// c.speak("I lost connection for a moment.")
	c.clearState()
}

func (c *Connector) getBlur() bool {
	resp := c.postJSON("/config", map[string]any{"get": true})
	if resp == nil {
		return true
	}
	conf, _ := resp["config"].(map[string]any)
	v, ok := conf["blur"].(bool)
	if !ok {
		return true
	}
	return v
}

func (c *Connector) setBlur(on bool) {
	_ = c.postJSON("/config", map[string]any{"set": map[string]any{"blur": on}})
}

func (c *Connector) waitAnyFace(ctx context.Context, timeoutSec int) bool {
	if timeoutSec <= 0 {
		timeoutSec = c.cfg.TimeoutSec
	}
	tries := (timeoutSec * 1000) / c.cfg.PollMs
	if tries < 1 {
		tries = 1
	}
	for i := 0; i < tries; i++ {
		resp := c.postJSON("/who", map[string]any{"recent_sec": c.cfg.FaceRecentSec})
		if resp != nil {
			now, _ := resp["now"].([]any)
			unknownNow, _ := resp["unknown_now"].(float64)
			if len(now)+int(unknownNow) >= 1 {
				return true
			}
		}
		select {
		case <-time.After(time.Duration(c.cfg.PollMs) * time.Millisecond):
		case <-ctx.Done():
			return false
		}
	}
	return false
}

// func displayName(id string) string {
// 	if id == "" {
// 		return ""
// 	}
// 	cleaned := dedupSuffixRE.ReplaceAllString(id, "")
// 	cleaned = strings.ReplaceAll(cleaned, "-", " ")
// 	cleaned = strings.ReplaceAll(cleaned, "_", " ")
// 	return cases.Title(language.English).String(strings.ToLower(cleaned))
// }

func shortUUID(uuid string) string {
	if len(uuid) >= 8 {
		return uuid[:8]
	}
	return uuid
}

// normID extracts a string arg and normalizes it into a valid identity id.
func normID(args map[string]any, k string) string {
	return normalizeID(util.StringFrom(args[k], ""))
}

// normalizeID maps arbitrary text onto the receiver's identity-name rules:
// lowercase, [a-z0-9_-], at most 32 characters.
func normalizeID(raw string) string {
	lowered := util.TrimLower(raw)

	var b strings.Builder
	lastDash := false
	for _, r := range lowered {
		switch {
		case r >= 'a' && r <= 'z', r >= '0' && r <= '9', r == '_':
			b.WriteRune(r)
			lastDash = false
		case r == ' ' || r == '-' || r == '\'' || r == '.':
			// Word separators collapse to a single dash, and never lead.
			if b.Len() > 0 && !lastDash {
				b.WriteRune('-')
				lastDash = true
			}
		default:
			// Accents, CJK, punctuation: dropped rather than transliterated.
			// A wrong guess at romanisation is worse than a shorter name,
			// and the model is asked for a Latin spelling in the prompt.
		}
	}

	out := strings.TrimRight(b.String(), "-")
	if len(out) > 32 {
		out = strings.TrimRight(out[:32], "-")
	}
	return out
}
