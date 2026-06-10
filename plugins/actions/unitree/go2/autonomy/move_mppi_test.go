package autonomy

import (
	"context"
	"encoding/binary"
	"math"
	"math/rand"
	"testing"
	"time"

	"go.uber.org/zap"

	zenohsession "github.com/openmind/om1/internal/zenoh"
)

type fakePublisher struct {
	puts    [][]byte
	dropped bool
	err     error
}

func (f *fakePublisher) Put(data []byte) error {
	f.puts = append(f.puts, append([]byte(nil), data...))
	return f.err
}

func (f *fakePublisher) Drop() { f.dropped = true }

type fakeSubscriber struct{ dropped bool }

func (f *fakeSubscriber) Drop() { f.dropped = true }

type fakeSession struct{ closed bool }

func (f *fakeSession) DeclarePublisher(string) (zenohsession.Publisher, error) {
	return nil, nil
}

func (f *fakeSession) DeclareSubscriber(string, func([]byte)) (zenohsession.Subscriber, error) {
	return nil, nil
}

func (f *fakeSession) Put(string, []byte) error { return nil }

func (f *fakeSession) Close() { f.closed = true }

// newTestMPPI builds a connector wired to a fake goal publisher, bypassing the
// real Zenoh session that the production constructor would open.
func newTestMPPI(pub *fakePublisher) *moveMPPIConnector {
	c := &moveMPPIConnector{
		log:          zap.NewNop(),
		goalPub:      pub,
		goalDistance: mppiGoalDistanceM,
		rng:          rand.New(rand.NewSource(1)),
	}
	c.aiControlEnabled.Store(true)
	return c
}

func readF64(b []byte, off int) float64 {
	return math.Float64frombits(binary.LittleEndian.Uint64(b[off:]))
}

// cdrString builds a CDR-encoded std_msgs/String payload matching what onStatus
// expects to decode at offset 4.
func cdrString(s string) []byte {
	b := []byte{0x00, 0x01, 0x00, 0x00}
	sb := append([]byte(s), 0x00)
	b = zenohsession.AppendUint32LE(b, uint32(len(sb)))
	return append(b, sb...)
}

func TestSerializePose(t *testing.T) {
	const yaw = math.Pi / 2
	payload := serializePose(1.5, -2.5, yaw, false)

	if len(payload) != 4+7*8 {
		t.Fatalf("pose length = %d, want %d", len(payload), 4+7*8)
	}
	if payload[0] != 0x00 || payload[1] != 0x01 || payload[2] != 0x00 || payload[3] != 0x00 {
		t.Errorf("unexpected CDR header: % x", payload[0:4])
	}

	if got := readF64(payload, 4); got != 1.5 {
		t.Errorf("position.x = %v, want 1.5", got)
	}
	if got := readF64(payload, 12); got != -2.5 {
		t.Errorf("position.y = %v, want -2.5", got)
	}
	if got := readF64(payload, 20); got != 0.0 {
		t.Errorf("position.z = %v, want 0 (not reverse)", got)
	}
	if got := readF64(payload, 44); math.Abs(got-math.Sin(yaw/2)) > 1e-12 {
		t.Errorf("orientation.z = %v, want %v", got, math.Sin(yaw/2))
	}
	if got := readF64(payload, 52); math.Abs(got-math.Cos(yaw/2)) > 1e-12 {
		t.Errorf("orientation.w = %v, want %v", got, math.Cos(yaw/2))
	}
}

func TestSerializePoseReverseFlag(t *testing.T) {
	payload := serializePose(-0.6, 0, 0, true)
	if got := readF64(payload, 20); got != 1.0 {
		t.Errorf("position.z = %v, want 1.0 (reverse flag set)", got)
	}
	if got := readF64(payload, 4); got != -0.6 {
		t.Errorf("position.x = %v, want -0.6", got)
	}
}

func TestPublishGoalAndReverse(t *testing.T) {
	pub := &fakePublisher{}
	c := newTestMPPI(pub)

	if err := c.publishGoal(1.2, 0, 0); err != nil {
		t.Fatalf("publishGoal failed: %v", err)
	}
	if err := c.publishReverseGoal(0.6); err != nil {
		t.Fatalf("publishReverseGoal failed: %v", err)
	}

	if len(pub.puts) != 2 {
		t.Fatalf("put count = %d, want 2", len(pub.puts))
	}
	if got := readF64(pub.puts[0], 4); got != 1.2 {
		t.Errorf("goal x = %v, want 1.2", got)
	}
	if got := readF64(pub.puts[1], 4); got != -0.6 {
		t.Errorf("reverse goal x = %v, want -0.6", got)
	}
	if got := readF64(pub.puts[1], 20); got != 1.0 {
		t.Errorf("reverse goal flag = %v, want 1.0", got)
	}
}

func TestPublishGoalNilPublisher(t *testing.T) {
	c := &moveMPPIConnector{log: zap.NewNop()}
	if err := c.publishGoal(1, 1, 1); err == nil {
		t.Error("expected error when publisher is nil")
	}
	if err := c.publishReverseGoal(1); err == nil {
		t.Error("expected error when publisher is nil")
	}
}

func TestActiveLifecycle(t *testing.T) {
	pub := &fakePublisher{}
	c := newTestMPPI(pub)

	c.markActive()
	c.mu.Lock()
	active, issued := c.active, c.issuedAt
	c.mu.Unlock()
	if !active {
		t.Fatal("expected active after markActive")
	}
	if issued.IsZero() {
		t.Error("expected issuedAt to be set")
	}

	c.cancelGoal()
	c.mu.Lock()
	active = c.active
	c.mu.Unlock()
	if active {
		t.Error("expected inactive after cancelGoal")
	}
	last := pub.puts[len(pub.puts)-1]
	if x, y := readF64(last, 4), readF64(last, 12); x != 0 || y != 0 {
		t.Errorf("cancel goal = (%v, %v), want origin", x, y)
	}
}

func TestIssueGoal(t *testing.T) {
	pub := &fakePublisher{}
	c := newTestMPPI(pub)

	c.issueGoal(nil, "advance")
	if len(pub.puts) != 0 {
		t.Fatalf("blocked move published %d goals, want 0", len(pub.puts))
	}
	c.mu.Lock()
	active := c.active
	c.mu.Unlock()
	if active {
		t.Error("expected inactive after blocked move")
	}

	// Path index 4 is straight ahead (0 deg): goal lies at goalDistance along +x.
	c.issueGoal([]uint32{4}, "advance")
	if len(pub.puts) != 1 {
		t.Fatalf("put count = %d, want 1", len(pub.puts))
	}
	if got := readF64(pub.puts[0], 4); math.Abs(got-c.goalDistance) > 1e-9 {
		t.Errorf("goal x = %v, want %v", got, c.goalDistance)
	}
	if got := readF64(pub.puts[0], 12); math.Abs(got) > 1e-9 {
		t.Errorf("goal y = %v, want 0", got)
	}
	c.mu.Lock()
	active = c.active
	c.mu.Unlock()
	if !active {
		t.Error("expected active after issuing goal")
	}
}

func TestIssueRetreat(t *testing.T) {
	pub := &fakePublisher{}
	c := newTestMPPI(pub)

	c.issueRetreat(false)
	if len(pub.puts) != 0 {
		t.Fatalf("blocked retreat published %d goals, want 0", len(pub.puts))
	}

	c.issueRetreat(true)
	if len(pub.puts) != 1 {
		t.Fatalf("put count = %d, want 1", len(pub.puts))
	}
	if got := readF64(pub.puts[0], 4); got != -mppiReverseDistanceM {
		t.Errorf("retreat x = %v, want %v", got, -mppiReverseDistanceM)
	}
	if got := readF64(pub.puts[0], 20); got != 1.0 {
		t.Errorf("retreat reverse flag = %v, want 1.0", got)
	}
}

func TestOnStatus(t *testing.T) {
	c := newTestMPPI(&fakePublisher{})

	for _, status := range []string{"reached", "idle", "blocked"} {
		c.mu.Lock()
		c.active = true
		c.mu.Unlock()

		c.onStatus(cdrString(status))

		c.mu.Lock()
		active := c.active
		c.mu.Unlock()
		if active {
			t.Errorf("status %q did not clear active", status)
		}
	}

	c.mu.Lock()
	c.active = true
	c.mu.Unlock()
	c.onStatus(cdrString("moving"))
	c.mu.Lock()
	active := c.active
	c.mu.Unlock()
	if !active {
		t.Error("non-terminal status should not clear active")
	}

	c.onStatus([]byte{0x00})
	c.mu.Lock()
	active = c.active
	c.mu.Unlock()
	if !active {
		t.Error("malformed status should not clear active")
	}
}

func TestOnAIStatusRequest(t *testing.T) {
	pub := &fakePublisher{}
	c := newTestMPPI(pub)
	c.aiRespPub = pub

	c.onAIStatusRequest(serializeAIStatusResponse("f", "r", aiCodeDisabled, "x"))
	if c.aiControlEnabled.Load() {
		t.Error("expected AI control disabled")
	}
	if len(pub.puts) != 1 {
		t.Fatalf("put count = %d, want 1", len(pub.puts))
	}

	c.onAIStatusRequest(serializeAIStatusResponse("f", "r", aiCodeEnabled, "x"))
	if !c.aiControlEnabled.Load() {
		t.Error("expected AI control enabled")
	}

	c.onAIStatusRequest(serializeAIStatusResponse("f", "r", aiCodeStatus, "x"))
	if !c.aiControlEnabled.Load() {
		t.Error("status request must not change control state")
	}
	if len(pub.puts) != 3 {
		t.Errorf("put count = %d, want 3", len(pub.puts))
	}

	before := len(pub.puts)
	c.onAIStatusRequest([]byte{0x00, 0x01})
	if len(pub.puts) != before {
		t.Errorf("malformed request published a response")
	}
}

func TestOnAIStatusRequestNilPublisher(t *testing.T) {
	c := &moveMPPIConnector{log: zap.NewNop()}
	c.aiControlEnabled.Store(true)
	c.onAIStatusRequest(serializeAIStatusResponse("f", "r", aiCodeDisabled, "x"))
	if !c.aiControlEnabled.Load() {
		t.Error("request must be ignored when no response publisher is available")
	}
}

func TestTick(t *testing.T) {
	pub := &fakePublisher{}
	c := newTestMPPI(pub)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	c.Tick(ctx)
	if len(pub.puts) != 0 {
		t.Errorf("cancelled Tick published %d goals, want 0", len(pub.puts))
	}

	pub = &fakePublisher{}
	c = newTestMPPI(pub)
	c.mu.Lock()
	c.active = true
	c.issuedAt = time.Now().Add(-commandTimeout - time.Second)
	c.mu.Unlock()
	c.Tick(context.Background())
	c.mu.Lock()
	active := c.active
	c.mu.Unlock()
	if active {
		t.Error("expected stale goal to be cleared")
	}
	if len(pub.puts) != 1 {
		t.Errorf("expected one stop goal on timeout, got %d", len(pub.puts))
	}

	pub = &fakePublisher{}
	c = newTestMPPI(pub)
	c.mu.Lock()
	c.active = true
	c.issuedAt = time.Now()
	c.mu.Unlock()
	c.Tick(context.Background())
	c.mu.Lock()
	active = c.active
	c.mu.Unlock()
	if !active {
		t.Error("fresh goal should not be cleared by Tick")
	}
	if len(pub.puts) != 0 {
		t.Errorf("fresh goal Tick published %d goals, want 0", len(pub.puts))
	}
}

func TestConnect(t *testing.T) {
	ctx := context.Background()

	c := newTestMPPI(&fakePublisher{})
	if _, err := c.Connect(ctx, 42); err == nil {
		t.Error("expected error for non-map input")
	}

	pub := &fakePublisher{}
	c = newTestMPPI(pub)
	c.aiControlEnabled.Store(false)
	if _, err := c.Connect(ctx, map[string]any{"action": "move forwards"}); err != nil {
		t.Fatalf("Connect returned error: %v", err)
	}
	if len(pub.puts) != 0 {
		t.Errorf("disabled control published %d goals, want 0", len(pub.puts))
	}

	pub = &fakePublisher{}
	c = newTestMPPI(pub)
	if _, err := c.Connect(ctx, map[string]any{"action": "stand still"}); err != nil {
		t.Fatalf("Connect returned error: %v", err)
	}
	if len(pub.puts) != 1 {
		t.Errorf("stand still published %d goals, want 1 (cancel)", len(pub.puts))
	}

	pub = &fakePublisher{}
	c = newTestMPPI(pub)
	c.mu.Lock()
	c.active = true
	c.mu.Unlock()
	if _, err := c.Connect(ctx, map[string]any{"action": "move forwards"}); err != nil {
		t.Fatalf("Connect returned error: %v", err)
	}
	if len(pub.puts) != 0 {
		t.Errorf("busy connector published %d goals, want 0", len(pub.puts))
	}

	// A single 'stand still' tick must not preempt a still-progressing recent
	// goal: at hertz=1 the LLM picks an action every tick and 'stand still'
	// shows up in several prompt examples.
	pub = &fakePublisher{}
	c = newTestMPPI(pub)
	c.minActiveHold = 3 * time.Second
	c.mu.Lock()
	c.active = true
	c.issuedAt = time.Now()
	c.mu.Unlock()
	if _, err := c.Connect(ctx, map[string]any{"action": "stand still"}); err != nil {
		t.Fatalf("Connect returned error: %v", err)
	}
	if len(pub.puts) != 0 {
		t.Errorf("stand still preempted fresh goal: %d goals, want 0", len(pub.puts))
	}
	c.mu.Lock()
	stillActive := c.active
	c.mu.Unlock()
	if !stillActive {
		t.Error("stand still cleared active flag on fresh goal")
	}

	// Two consecutive 'stand still' ticks express a sustained intent and DO
	// cancel, regardless of how recently the goal was issued.
	pub = &fakePublisher{}
	c = newTestMPPI(pub)
	c.minActiveHold = 3 * time.Second
	c.lastAction = "stand still"
	c.mu.Lock()
	c.active = true
	c.issuedAt = time.Now()
	c.mu.Unlock()
	if _, err := c.Connect(ctx, map[string]any{"action": "stand still"}); err != nil {
		t.Fatalf("Connect returned error: %v", err)
	}
	if len(pub.puts) != 1 {
		t.Errorf("repeated stand still failed to cancel: %d goals, want 1", len(pub.puts))
	}

	pub = &fakePublisher{}
	c = newTestMPPI(pub)
	if _, err := c.Connect(ctx, map[string]any{"action": "stand still"}); err != nil {
		t.Fatalf("Connect returned error: %v", err)
	}
	if len(pub.puts) != 1 {
		t.Errorf("idle stand still: %d goals, want 1 (cancel)", len(pub.puts))
	}
	if c.lastAction != "stand still" {
		t.Errorf("lastAction = %q, want \"stand still\"", c.lastAction)
	}
}

func TestStopReleasesResources(t *testing.T) {
	goalPub := &fakePublisher{}
	aiRespPub := &fakePublisher{}
	statusSub := &fakeSubscriber{}
	aiReqSub := &fakeSubscriber{}
	sess := &fakeSession{}

	c := &moveMPPIConnector{
		log:       zap.NewNop(),
		goalPub:   goalPub,
		aiRespPub: aiRespPub,
		statusSub: statusSub,
		aiReqSub:  aiReqSub,
		session:   sess,
	}

	c.Stop()

	if !goalPub.dropped || !aiRespPub.dropped {
		t.Error("expected publishers to be dropped")
	}
	if !statusSub.dropped || !aiReqSub.dropped {
		t.Error("expected subscribers to be dropped")
	}
	if !sess.closed {
		t.Error("expected session to be closed")
	}
	if c.goalPub != nil || c.aiRespPub != nil || c.statusSub != nil || c.aiReqSub != nil || c.session != nil {
		t.Error("expected all Zenoh handles to be cleared")
	}

	c.Stop()
}
