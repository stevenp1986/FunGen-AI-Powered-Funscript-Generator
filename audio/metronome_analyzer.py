import numpy as np
from collections import deque
from typing import Optional, Tuple


class AudioBeatAnalyzer:
    """
    High-quality, drift-free audio beat/novelty analyzer.
    - Uses the app's VideoProcessor audio envelope (extracted via FFmpeg) to avoid drift.
    - Maintains EMA baseline, novelty history, and derivative-based onset measure.
    - Provides robust z-score using median/MAD for stability.
    """

    def __init__(self, app, env_history_seconds: float = 2.0, novelty_history_len: int = 200):
        self.app = app
        self.env_history_seconds = max(0.5, float(env_history_seconds))
        self.novelty_history_len = max(50, int(novelty_history_len))

        # Runtime state
        self.ema: Optional[float] = None
        self.ema_alpha: float = self._get_setting('beat_audio_ema_alpha', 0.2, min_val=0.01, max_val=0.9)
        self.env_history = deque(maxlen=int(2.0 * 200))  # Keep as float envelope values (not strictly needed)
        self.novelty_history = deque(maxlen=self.novelty_history_len)
        self.last_novelty: Optional[float] = None

    def _get_setting(self, key: str, default: float, min_val: Optional[float] = None, max_val: Optional[float] = None) -> float:
        try:
            get = self.app.app_settings.get if (self.app and hasattr(self.app, 'app_settings')) else (lambda k, d=None: d)
            val = float(get(key, default))
            if min_val is not None:
                val = max(min_val, val)
            if max_val is not None:
                val = min(max_val, val)
            return val
        except Exception:
            return default

    def reset(self):
        self.ema = None
        self.novelty_history.clear()
        self.env_history.clear()
        self.last_novelty = None

    def update(self, media_time_ms: float) -> Tuple[float, float, float, float, float]:
        """
        Update analyzer at a given media time (ms).
        Returns a tuple: (signal, novelty, novelty_deriv, z, z_deriv)
        - signal: raw envelope (0..1)
        - novelty: max(0, signal - EMA)
        - novelty_deriv: first difference of novelty
        - z: robust z-score of novelty
        - z_deriv: robust z-score of novelty derivative
        """
        # Pull envelope value from VideoProcessor (pre-extracted, drift-free)
        processor = getattr(self.app, 'processor', None)
        signal = 0.0
        if processor is not None:
            try:
                val = processor.get_audio_envelope_value(media_time_ms)
                if val is not None:
                    signal = float(val)
            except Exception:
                signal = 0.0
        # EMA baseline
        if self.ema is None:
            self.ema = signal
        else:
            self.ema = (1.0 - self.ema_alpha) * float(self.ema) + self.ema_alpha * signal
        novelty = max(0.0, signal - float(self.ema))
        # Derivative of novelty
        if self.last_novelty is None:
            novelty_deriv = 0.0
        else:
            novelty_deriv = float(novelty) - float(self.last_novelty)
        self.last_novelty = float(novelty)

        # Update histories
        self.env_history.append(signal)
        self.novelty_history.append(float(novelty))

        # Robust stats on novelty
        z = 0.0
        z_deriv = 0.0
        try:
            nov = np.array(self.novelty_history, dtype=float)
            if nov.size >= 3:
                med = float(np.median(nov))
                mad = float(np.median(np.abs(nov - med)))
                scale = (1.4826 * mad) + 1e-6
                z = (float(novelty) - med) / scale
                # Derivative robust stats
                d = np.diff(nov)
                if d.size >= 8:
                    dmed = float(np.median(d))
                    dmad = float(np.median(np.abs(d - dmed)))
                    dscale = (1.4826 * dmad) + 1e-6
                else:
                    dscale = scale
                z_deriv = float(novelty_deriv) / dscale
        except Exception:
            z = 0.0
            z_deriv = 0.0

        return signal, float(novelty), float(novelty_deriv), float(z), float(z_deriv)
