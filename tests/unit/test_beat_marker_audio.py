import numpy as np
import cv2
import types
import pytest
import sys, os

# Make project root importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Stub ultralytics if not installed to allow importing ROITracker without heavy deps
if 'ultralytics' not in sys.modules:
    sys.modules['ultralytics'] = types.SimpleNamespace(YOLO=object)

from tracker.tracker import ROITracker


class FakeProcessor:
    def __init__(self, envelope_series, fps=30.0):
        self._envelope_series = envelope_series
        self._index = 0
        self.fps = fps
        # Attributes referenced by ROITracker when writing actions
        self.current_frame_index = 0
        self.video_info = {'fps': fps}

    def get_audio_envelope_value(self, time_ms: int) -> float:
        # Return based on time_ms mapped to index by 10ms per step
        idx = min(int(round(time_ms / 10.0)), len(self._envelope_series) - 1)
        return float(self._envelope_series[idx])


class FakeApp:
    def __init__(self, envelope_series, beat_threshold_sigma=1.5, hyster_ratio=0.6, min_interval_ms=120):
        self.logger = types.SimpleNamespace(info=lambda *a, **k: None, debug=lambda *a, **k: None, warning=lambda *a, **k: None, error=lambda *a, **k: None)
        self.processor = FakeProcessor(envelope_series)
        self.tracker = types.SimpleNamespace(
            output_delay_frames=0,
            current_video_fps_for_delay=self.processor.fps,
            y_offset=0.0,
            x_offset=0.0,
            sensitivity=70.0,
        )
        # Axis selection used when writing funscript
        self.tracking_axis_mode = 'both'
        self.single_axis_output_target = 'primary'
        # Minimal app_settings to drive Beat Marker
        self.app_settings = {
            'beat_source': 'audio',
            'beat_bpm': 0,  # disable metronome adjustments of min_interval
            'beat_subdivision': 1,
            'beat_amp_min': 10,
            'beat_amp_max': 90,
            'beat_waveform': 'step',
            'beat_threshold_sigma': beat_threshold_sigma,
            'beat_hysteresis_ratio': hyster_ratio,
            'beat_min_interval_ms': min_interval_ms,
            'beat_swing_percent': 0.0,
            'beat_phase_deg': 0.0,
        }


def make_black_frame(w=320, h=240):
    return np.zeros((h, w, 3), dtype=np.uint8)


@pytest.mark.parametrize("spike_vals", [
    # Simple baseline then 50ms spike, drop, then another 50ms spike
    ([0.05]*20 + [0.9]*5 + [0.05]*20 + [0.95]*5 + [0.05]*20),
])
def test_audio_beat_triggers_on_envelope_spikes(spike_vals):
    app = FakeApp(spike_vals, beat_threshold_sigma=1.2, hyster_ratio=0.6, min_interval_ms=120)
    tracker = ROITracker(app_logic_instance=app, tracker_model_path="", load_models_on_init=False)

    # Ensure Beat Marker mode and start state
    tracker.set_tracking_mode("BEAT_MARKER")
    tracker.start_tracking()

    # Feed frames at 30 FPS; frame_time_ms increments ~33ms; we map envelope sampling by 10ms index resolution
    frame = make_black_frame()
    action_events = []
    frame_time_ms = 0
    for i in range(0, 300):
        processed, actions = tracker.process_frame_for_beat_marker(frame, frame_time_ms, frame_index=i)
        if actions:
            action_events.extend(actions)
        frame_time_ms += 33  # ~30 fps

    # We expect at least 2 triggers matching two spikes
    assert len(action_events) >= 2, f"Expected >=2 audio beat triggers, got {len(action_events)}"

    # Ensure timestamps respect min_interval gating
    stamps = [a['at'] for a in action_events]
    deltas = [stamps[i] - stamps[i-1] for i in range(1, len(stamps))]
    assert all(d >= app.app_settings['beat_min_interval_ms'] - 10 for d in deltas), f"Interval gating failed: {deltas}"


def test_audio_no_trigger_without_spike():
    # Flat low envelope should not trigger
    vals = [0.05] * 200
    app = FakeApp(vals, beat_threshold_sigma=1.5, hyster_ratio=0.6, min_interval_ms=120)
    tracker = ROITracker(app_logic_instance=app, tracker_model_path="", load_models_on_init=False)
    tracker.set_tracking_mode("BEAT_MARKER")
    tracker.start_tracking()

    frame = make_black_frame()
    frame_time_ms = 0
    events = []
    for i in range(0, 200):
        _, actions = tracker.process_frame_for_beat_marker(frame, frame_time_ms, frame_index=i)
        if actions:
            events.extend(actions)
        frame_time_ms += 33

    assert len(events) == 0, f"Unexpected triggers on flat envelope: {len(events)}"
