import threading
import time
import queue
from typing import Optional
import numpy as np

class SoundDevicePyAVAudioBackend:
    """
    Minimal audio playback backend using PyAV (decode) + sounddevice (output).
    Designed to be resilient if dependencies are missing; it will disable itself
    gracefully and only log warnings.
    """
    def __init__(self, logger):
        self.logger = logger
        self._available = False
        try:
            import sounddevice as sd  # type: ignore
            import av  # type: ignore
            self._sd = sd
            self._av = av
            self._available = True
        except Exception as e:
            self._err = f"Audio backend unavailable: {e}"
            self._available = False
        self._stream = None
        self._worker: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._q: queue.Queue = queue.Queue(maxsize=12)
        self._vol = 1.0
        self._running = False
        self._cb_buf = None  # type: Optional[np.ndarray]

    @property
    def available(self) -> bool:
        return self._available

    def set_volume(self, volume_linear: float) -> None:
        self._vol = max(0.0, min(1.0, float(volume_linear)))

    def is_running(self) -> bool:
        return self._running

    def stop(self) -> None:
        try:
            self._stop_evt.set()
            if self._worker and self._worker.is_alive():
                self._worker.join(timeout=1.0)
        except Exception:
            pass
        self._worker = None
        try:
            if self._stream:
                self._stream.stop()
                self._stream.close()
        except Exception:
            pass
        self._stream = None
        try:
            while not self._q.empty():
                self._q.get_nowait()
        except Exception:
            pass
        self._running = False

    def start(self, media_path: str, start_time_s: float) -> None:
        """
        Start playback from media_path at the given start time (seconds).
        If not available or error occurs, it will log and do nothing.
        """
        if not self._available:
            if self._err:
                self.logger.info(self._err)
            return
        self.stop()
        try:
            container = self._av.open(media_path, mode='r')
            stream = next((s for s in container.streams if s.type == 'audio'), None)
            if stream is None:
                self.logger.warning("Audio backend: no audio stream in media.")
                container.close()
                return
            # Target sample rate: force 48000 Hz for macOS device compatibility
            target_rate = 48000
            target_channels = 2  # force stereo output for simplicity
            # Seek using stream.time_base for precise alignment
            try:
                if stream.time_base:
                    seek_ts = int(max(0.0, start_time_s) / float(stream.time_base))
                    container.seek(seek_ts, any_frame=True, backward=False, stream=stream)
            except Exception:
                pass

            sd = self._sd
            av = self._av

            # Prepare a resampler to a consistent float32 planar format at 48k
            try:
                # 'fltp' = float32 planar; we'll interleave to [samples, channels]
                resampler = av.audio.resampler.AudioResampler(format='fltp', layout='stereo', rate=target_rate)
            except Exception as e:
                self.logger.warning(f"Audio backend: failed to create resampler: {e}")
                container.close()
                return

            def callback(outdata, frames, time_info, status):
                if status.output_underflow:
                    # Fill with silence on underflow
                    outdata[:] = 0
                    return
                # Ensure we have a carry-over buffer
                if self._cb_buf is None:
                    self._cb_buf = np.empty((0, outdata.shape[1]), dtype=np.float32)
                buf = self._cb_buf
                # Pull from queue until enough frames or queue empty
                while buf.shape[0] < frames:
                    try:
                        nxt = self._q.get_nowait()
                    except queue.Empty:
                        break
                    if nxt.dtype != np.float32:
                        nxt = nxt.astype(np.float32, copy=False)
                    # Apply volume then hard-clip to avoid overdrive
                    if self._vol != 1.0:
                        nxt *= self._vol
                    # Clip regardless, in case upstream amplitude exceeds 1.0
                    np.clip(nxt, -1.0, 1.0, out=nxt)
                    # Ensure channels match
                    if nxt.ndim == 1:
                        nxt = nxt.reshape(-1, 1)
                    if nxt.shape[1] != outdata.shape[1]:
                        if nxt.shape[1] < outdata.shape[1]:
                            nxt = np.repeat(nxt, outdata.shape[1], axis=1)[:, :outdata.shape[1]]
                        else:
                            nxt = nxt[:, :outdata.shape[1]]
                    buf = np.concatenate([buf, nxt], axis=0)
                # Output exactly 'frames' samples
                if buf.shape[0] >= frames:
                    outdata[:] = buf[:frames, :]
                    self._cb_buf = buf[frames:, :]
                else:
                    outdata[:buf.shape[0], :] = buf
                    if buf.shape[0] < frames:
                        outdata[buf.shape[0]:, :] = 0
                    self._cb_buf = np.empty((0, outdata.shape[1]), dtype=np.float32)

            self._stream = sd.OutputStream(
                samplerate=target_rate,
                channels=target_channels,
                dtype='float32',
                callback=callback,
                # Use a moderate block size for stability across devices
                blocksize=512,
            )

            def worker():
                try:
                    # Clear any stale buffered audio
                    try:
                        while True:
                            self._q.get_nowait()
                    except Exception:
                        pass
                    # Prebuffer ~100ms of audio before starting the stream to avoid initial underflow
                    prebuffer_target_frames = int(0.10 * target_rate)
                    queued_frames = 0
                    for packet in container.demux(stream):
                        if self._stop_evt.is_set():
                            break
                        for frame in packet.decode():
                            if self._stop_evt.is_set():
                                break
                            # Resample to s16/stereo/target_rate; resample may return a frame or list
                            try:
                                resampled = resampler.resample(frame)
                            except Exception:
                                resampled = None
                            if not resampled:
                                continue
                            frames_out = resampled if isinstance(resampled, (list, tuple)) else [resampled]
                            for out_f in frames_out:
                                # Skip frames earlier than requested start time (extra safety if seek landed early)
                                try:
                                    pts = out_f.pts
                                    tb = out_f.time_base or stream.time_base
                                    if pts is not None and tb:
                                        ftime = float(pts) * float(tb)
                                        if ftime + 0.010 < max(0.0, start_time_s):
                                            continue
                                except Exception:
                                    pass
                                try:
                                    arr = out_f.to_ndarray()  # float32 planar or packed depending on codec/resampler
                                except Exception:
                                    continue
                                # Determine orientation: expect either [channels, samples] or [samples, channels]
                                if arr.ndim == 1:
                                    # mono packed -> [samples]
                                    arr = arr.reshape(-1, 1)  # [samples, 1]
                                else:
                                    ch_guess_0 = arr.shape[0]
                                    ch_guess_1 = arr.shape[1] if arr.ndim > 1 else 1
                                    if ch_guess_0 in (1, 2) and ch_guess_1 not in (1, 2):
                                        # [channels, samples] -> transpose to [samples, channels]
                                        arr = arr.T
                                    elif ch_guess_1 in (1, 2) and ch_guess_0 not in (1, 2):
                                        # already [samples, channels]
                                        pass
                                    else:
                                        # Fallback: if second dim is shorter, assume it's channels
                                        if ch_guess_1 <= ch_guess_0:
                                            pass  # [samples, channels]
                                        else:
                                            arr = arr.T
                                # Ensure float32 in [-1, 1]
                                if arr.dtype != np.float32:
                                    arr = arr.astype(np.float32, copy=False)
                                    # fltp should already be float32; if not, normalize as best-effort
                                    max_abs = np.max(np.abs(arr)) if arr.size else 1.0
                                    scale = (1.0 / 32768.0) if max_abs > 2.0 else 1.0
                                    arr *= scale
                                # Ensure exactly 2 channels
                                if arr.shape[1] < target_channels:
                                    arr = np.repeat(arr, target_channels, axis=1)[:, :target_channels]
                                elif arr.shape[1] > target_channels:
                                    arr = arr[:, :target_channels]
                                pcm = arr
                                # Split into moderate chunks for smoother callback consumption
                                block = 1024
                                total = pcm.shape[0]
                                pos = 0
                                try:
                                    while pos < total:
                                        end = min(pos + block, total)
                                        self._q.put(pcm[pos:end, :], timeout=0.5)
                                        # Track prebuffer fill
                                        queued_frames += (end - pos)
                                        pos = end
                                except queue.Full:
                                    # Drop if output can't keep up
                                    pass
                            # Start the stream once we've prebuffered enough (one-time)
                            if not self._running and queued_frames >= prebuffer_target_frames:
                                try:
                                    self._stream.start()
                                    self._running = True
                                except Exception:
                                    # If start fails, try again next iteration
                                    pass
                    # If demux loop ended without starting (short file), try to start anyway
                    if not self._running:
                        try:
                            self._stream.start()
                            self._running = True
                        except Exception:
                            pass
                except Exception as e:
                    self.logger.warning(f"Audio backend worker error: {e}")
                finally:
                    try:
                        container.close()
                    except Exception:
                        pass
                    self._running = False

            self._stop_evt.clear()
            self._worker = threading.Thread(target=worker, name="AudioSDPyAVWorker", daemon=True)
            self._worker.start()
        except Exception as e:
            self.logger.warning(f"Audio backend start failed: {e}")
            self.stop()
