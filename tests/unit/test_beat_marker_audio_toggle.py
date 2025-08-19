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
        self.fps = fps
        # Attributes referenced by ROITracker when writing actions
        self.current_frame_index = 0
        self.video_info = {'fps': fps}

    def get_audio_envelope_value(self, time_ms: int) -> float:
        # Return based on time_ms mapped to index by 10ms per step
        idx = min(int(round(time_ms / 10.0)), len(self._envelope_series) - 1)
        return float(self._envelope_series[idx])


class FakeApp:
    def __init__(self, envelope_series,
                 beat_threshold_sigma=1.2, hyster_ratio=0.6,
                 min_interval_ms=120, amp_min=20, amp_max=80, waveform='step'):
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
            'beat_amp_min': amp_min,
            'beat_amp_max': amp_max,
            'beat_waveform': waveform,
            'beat_threshold_sigma': beat_threshold_sigma,
            'beat_hysteresis_ratio': hyster_ratio,
            'beat_min_interval_ms': min_interval_ms,
            'beat_swing_percent': 0.0,
            'beat_phase_deg': 0.0,
        }


def make_black_frame(w=320, h=240):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_audio_amp_toggles_min_max_step_waveform():
    # Envelope: baseline, spike, baseline, spike, baseline, spike ...
    spikes = ([0.05]*20 + [0.95]*5 + [0.05]*20 + [0.9]*5 + [0.05]*20 + [0.92]*5 + [0.05]*20)
    app = FakeApp(spikes, amp_min=20, amp_max=80, waveform='step', beat_threshold_sigma=1.2)
    tracker = ROITracker(app_logic_instance=app, tracker_model_path="", load_models_on_init=False)

    # Ensure Beat Marker mode and start state
    tracker.set_tracking_mode("BEAT_MARKER")
    tracker.start_tracking()

    frame = make_black_frame()
    frame_time_ms = 0
    events = []
    for i in range(0, 400):
        _, actions = tracker.process_frame_for_beat_marker(frame, frame_time_ms, frame_index=i)
        if actions:
            events.extend(actions)
        frame_time_ms += 33  # ~30fps

    # Need at least 3 toggles
    assert len(events) >= 3, f"Expected >=3 beat events, got {len(events)}"

    # Extract positions written and verify alternation 80,20,80,20,... starting from amp_max
    positions = [e['pos'] for e in events[:5]]
    for idx in range(1, len(positions)):
        assert positions[idx] != positions[idx-1], f"Positions did not alternate: {positions}"
        assert positions[idx] in (20, 80), f"Unexpected amplitude {positions[idx]} not in range [20,80]"


def test_audio_amp_clamped_and_swapped_if_needed():
    # Set min > max and out-of-range values; code should clamp and swap
    spikes = ([0.05]*20 + [0.95]*5 + [0.05]*20 + [0.95]*5 + [0.05]*20)
    app = FakeApp(spikes, amp_min=120, amp_max=-10, waveform='step', beat_threshold_sigma=1.2)
    tracker = ROITracker(app_logic_instance=app, tracker_model_path="", load_models_on_init=False)
    tracker.set_tracking_mode("BEAT_MARKER")
    tracker.start_tracking()

    frame = make_black_frame()
    frame_time_ms = 0
    events = []
    for i in range(0, 250):
        _, actions = tracker.process_frame_for_beat_marker(frame, frame_time_ms, frame_index=i)
        if actions:
            events.extend(actions)
        frame_time_ms += 33

    assert len(events) >= 2
    # After clamping and swap, expected amps should be 0 and 100
    positions = [e['pos'] for e in events[:2]]
    assert set(positions).issubset({0, 100}), f"Clamped/swap positions should be 0/100, got {positions}"
