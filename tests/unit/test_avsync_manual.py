import os
import time
import logging
import re

import pytest

from video.video_processor import VideoProcessor


VIDEO_PATH = "/Users/steven/dwhelper/Dildo Hero - Riding The A Train - Videos - Hypnotube.mp4"
TARGET_START_S = 5 * 60 + 47  # 347s
PROCESS_SECONDS = 30
TIMEOUT_S = 120


class MockApp:
    def __init__(self):
        self.logger = logging.getLogger("test")
        self.logger.setLevel(logging.DEBUG)
        # Use dict-like settings
        self.app_settings = {
            # Stabilize drift controller for testing
            "audio_resync_min_interval_s": 5.0,
            "audio_drift_required_consecutive": 6,
            "audio_drift_resync_ms": 250,
            "audio_drift_check_interval_ms": 300,
            # Prefer index-based start to remove PTS variability during initial sync
            "audio_sync_use_pts": False,
            # Ensure audio plays so we can test A/V behaviour
            "audio_playback_enabled": True,
            # Default volume
            "audio_volume": 0,
        }
        # Minimal attributes referenced by VideoProcessor
        self.hardware_acceleration_method = "none"
        self.available_ffmpeg_hwaccels = []
        # Provide a minimal file_manager mock used by open_video()
        class _FM:
            def __init__(self):
                self.preprocessed_video_path = None

            def get_output_path_for_file(self, src_path: str, suffix: str):
                # For tests, pretend there is no preprocessed file
                # Return a non-existent but valid string path
                return "/__no_preprocessed__.mkv"

        self.file_manager = _FM()


@pytest.mark.skipif(not os.path.exists(VIDEO_PATH), reason="Test video not found on this machine")
def test_avsync_seek_and_run_short_window(caplog):
    caplog.set_level(logging.DEBUG)

    app = MockApp()
    vp = VideoProcessor(app_instance=app)

    assert vp.open_video(VIDEO_PATH) is True
    assert vp.is_video_open() is True

    fps = float(vp.video_info.get("fps", 30.0)) or 30.0
    start_frame = int(fps * TARGET_START_S)
    end_frame = start_frame + int(fps * PROCESS_SECONDS)

    # Start processing the short window
    vp.start_processing(start_frame=start_frame, end_frame=end_frame)

    # Wait until processing thread exits or timeout
    t0 = time.time()
    while vp.processing_thread and vp.processing_thread.is_alive():
        if time.time() - t0 > TIMEOUT_S:
            break
        time.sleep(0.05)

    # Stop/cleanup defensively
    vp.stop_processing(join_thread=True)
    # Extra safety: ensure audio is stopped even if processing loop exited unexpectedly
    if hasattr(vp, "_stop_audio_playback"):
        try:
            vp._stop_audio_playback()
        except Exception:
            pass

    # Gather logs
    logs = "\n".join(rec.message for rec in caplog.records if rec.name.endswith("app_logic") or rec.name.endswith("video_processor"))

    # Extract drift lines
    drift_lines = [line for line in logs.splitlines() if "A/V drift detected" in line]

    # Report counts to help tuning
    print(f"Drift events: {len(drift_lines)}")
    for line in drift_lines[:5]:
        print(line)

    # Expect limited resyncs in a 30s window. Heuristic to catch thrashing
    assert len(drift_lines) <= 4, f"Too many drift resyncs ({len(drift_lines)}). Check pacing and thresholds."
