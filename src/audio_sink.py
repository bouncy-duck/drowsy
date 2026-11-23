from __future__ import annotations
from pathlib import Path

try:
    import winsound
    _HAS_WINSOUND = True
except ImportError:
    winsound = None
    _HAS_WINSOUND = False

from src.alarm_policy import AlarmLevel


class WinsoundSink:
    """
    Windows winsound sink. One looping WAV per level.

    Level swap is atomic: calling PlaySound with a new file replaces the
    currently playing sound. Call only on transitions (set() early-returns
    if the level is unchanged, so it is safe to call every frame).
    """

    def __init__(self, mild_path, intense_path, face_lost_path):
        self._paths = {
            AlarmLevel.MILD:      str(Path(mild_path)),
            AlarmLevel.INTENSE:   str(Path(intense_path)),
            AlarmLevel.FACE_LOST: str(Path(face_lost_path)),
        }
        self._current = AlarmLevel.IDLE

    def validate(self):
        if not _HAS_WINSOUND:
            raise RuntimeError(
                "winsound is unavailable — this feature requires Windows."
            )
        missing = [p for p in self._paths.values() if not Path(p).is_file()]
        if missing:
            raise FileNotFoundError(
                "Alarm WAV file(s) missing:\n  " + "\n  ".join(missing)
            )

    def set(self, level: AlarmLevel):
        level = AlarmLevel(level)
        if level == self._current:
            return
        if level == AlarmLevel.IDLE:
            self._stop()
            return
        winsound.PlaySound(
            self._paths[level],
            winsound.SND_FILENAME | winsound.SND_ASYNC
            | winsound.SND_LOOP | winsound.SND_NODEFAULT,
        )
        self._current = level

    def _stop(self):
        try:
            winsound.PlaySound(None, winsound.SND_PURGE)
        except Exception:
            # Documented SND_PURGE caveat on modern Windows; silent fallback.
            pass
        self._current = AlarmLevel.IDLE

    def stop(self):
        self._stop()


class NullSink:
    """No-op sink (--no-alarm, or for platforms without winsound)."""
    def __init__(self):
        self.current = AlarmLevel.IDLE
    def set(self, level):
        self.current = AlarmLevel(level)
    def stop(self):
        self.current = AlarmLevel.IDLE
    def validate(self):
        pass


class RecordingSink:
    """For unit tests: records every level transition."""
    def __init__(self):
        self.events = []
        self.current = AlarmLevel.IDLE
    def set(self, level):
        level = AlarmLevel(level)
        if level != self.current:
            self.events.append(level)
            self.current = level
    def stop(self):
        self.set(AlarmLevel.IDLE)
    def validate(self):
        pass