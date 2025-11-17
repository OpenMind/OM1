from providers.face_presence_provider import FacePresenceProvider, PresenceSnapshot


class _FakeResp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if not (200 <= self.status_code < 300):
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def _patch_post(monkeypatch, provider: FacePresenceProvider, payload: dict):
    """Patch requests.Session.post to return a fake /who response."""

    def _fake_post(url, json, timeout):
        # mimic what /who would return
        return _FakeResp(payload)

    monkeypatch.setattr(provider._session, "post", _fake_post)


def test_presence_snapshot_to_text_variants():
    # 1 known, 0 unknown
    snap = PresenceSnapshot(ts=1.0, names=["wendy"], unknown_faces=0, raw={})
    assert snap.to_text() == "In Camera View: 1 known (wendy)."

    # 3 known, 2 unknown (ordering and 'and' grammar)
    snap = PresenceSnapshot(
        ts=1.0, names=["wendy", "alice", "bob"], unknown_faces=2, raw={}
    )
    assert (
        snap.to_text()
        == "In Camera View: 3 known (wendy, alice and bob) and 2 unknown faces."
    )

    # 0 known, 1 unknown (singular form)
    snap = PresenceSnapshot(ts=1.0, names=[], unknown_faces=1, raw={})
    assert snap.to_text() == "In Camera View: 1 unknown face."

    # duplicates and 'unknown' should be cleaned out
    snap = PresenceSnapshot(
        ts=1.0, names=["wendy", "wendy", "unknown"], unknown_faces=0, raw={}
    )
    assert snap.to_text() == "In Camera View: 1 known (wendy)."

    # nothing
    snap = PresenceSnapshot(ts=1.0, names=[], unknown_faces=0, raw={})
    assert snap.to_text() == "No one in view."


# ---------------------- frames-based suppression ---------------------- #
def test_frames_based_unknown_suppressed(monkeypatch):
    """
    Suppress when frames_recent >= min_obs_window and
    frames_with_unknown / frames_recent < unknown_frac_threshold.
    """
    provider = FacePresenceProvider(
        base_url="http://fake",
        recent_sec=3.0,
        unknown_frac_threshold=0.15,  # 15%
        min_obs_window=24,  # need at least 24 frames
    )
    provider.prefer_recent = True

    payload = {
        "server_ts": 1000.0,
        "recent_sec": 3.0,
        "now": ["wendy"],
        "unknown_now": 0,
        "frames_recent": 40,  # >= min_obs_window
        "frames_with_unknown": 4,  # 4 / 40 = 10% < 15%
        "recent_name_frames": {"wendy": 30},
        "unknown_recent": 3,  # peak unknown in any frame
    }
    _patch_post(monkeypatch, provider, payload)

    snap = provider._fetch_snapshot()
    assert set(snap.names) == {"wendy"}
    assert snap.unknown_faces == 0
    assert provider.unknown_faces == 0


def test_frames_based_unknown_kept(monkeypatch):
    """
    Keep peak when frequency >= threshold (or window too small).
    """
    provider = FacePresenceProvider(
        base_url="http://fake",
        recent_sec=3.0,
        unknown_frac_threshold=0.15,
        min_obs_window=24,
    )
    provider.prefer_recent = True

    payload = {
        "server_ts": 1000.0,
        "recent_sec": 3.0,
        "now": ["wendy"],
        "unknown_now": 0,
        "frames_recent": 40,
        "frames_with_unknown": 20,  # 50% >= 15%
        "recent_name_frames": {"wendy": 28},
        "unknown_recent": 5,  # peak unknown per frame
    }
    _patch_post(monkeypatch, provider, payload)

    snap = provider._fetch_snapshot()
    assert set(snap.names) == {"wendy"}
    assert snap.unknown_faces == 5
    assert provider.unknown_faces == 5


def test_frames_recent_zero_falls_back_to_now(monkeypatch):
    """
    When frames_recent == 0, provider should fall back to `now` values.
    """
    provider = FacePresenceProvider(base_url="http://fake", recent_sec=3.0)
    provider.prefer_recent = True

    payload = {
        "server_ts": 1000.0,
        "recent_sec": 3.0,
        "now": ["wendy", "unknown", "unknown"],
        "unknown_now": 2,
        "frames_recent": 0,
        "frames_with_unknown": 0,
        "recent_name_frames": {},
        "unknown_recent": 0,
    }
    _patch_post(monkeypatch, provider, payload)

    snap = provider._fetch_snapshot()
    assert set(snap.names) == {"wendy"}
    assert snap.unknown_faces == 2
    assert provider.unknown_faces == 2


def test_prefer_now_path(monkeypatch):
    """
    prefer_recent=False → use `now` path regardless of frames fields.
    """
    provider = FacePresenceProvider(base_url="http://fake", recent_sec=3.0)
    provider.prefer_recent = False

    payload = {
        "server_ts": 1000.0,
        "recent_sec": 3.0,
        "now": ["alice", "alice", "unknown"],
        "unknown_now": 1,
        "frames_recent": 999,
        "frames_with_unknown": 999,
        "recent_name_frames": {"ignored": 999},
        "unknown_recent": 99,
    }
    _patch_post(monkeypatch, provider, payload)

    snap = provider._fetch_snapshot()
    assert set(snap.names) == {"alice"}
    assert snap.unknown_faces == 1
    assert provider.unknown_faces == 1


# ---------------------- plumbing / misc ---------------------- #
def test_set_recent_sec_clamps():
    provider = FacePresenceProvider(base_url="http://fake", recent_sec=3.0)
    provider.set_recent_sec(-10)
    assert provider.recent_sec == 0.0
    provider.set_recent_sec(2.5)
    assert provider.recent_sec == 2.5


def test_emit_invokes_callbacks(monkeypatch):
    provider = FacePresenceProvider(base_url="http://fake", recent_sec=3.0)
    got = []

    def cb(line: str):
        got.append(line)

    provider.register_message_callback(cb)
    snap = PresenceSnapshot(ts=1.0, names=["alice", "bob"], unknown_faces=1, raw={})
    provider._emit(snap.to_text())

    assert got, "callback should be invoked"
    assert "In Camera View: 2 known (alice and bob) and 1 unknown face." in got[0]
