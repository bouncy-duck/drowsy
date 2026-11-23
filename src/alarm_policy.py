from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional


class AlarmLevel(IntEnum):
    IDLE      = 0
    MILD      = 1
    INTENSE   = 2
    FACE_LOST = 3


@dataclass
class AlarmState:
    level: AlarmLevel
    drowsy_run_s: float
    non_drowsy_run_s: float
    face_lost_sec: float
    face_lost_latched: bool
    mild_latched: bool
    intense_latched: bool
    initial_warmup_done: bool


class AlarmPolicy:
    """
    Pure state machine. No I/O, no clock source inside; `now` is passed in
    so this is unit-testable with a fake clock.

    Inputs per update():
      now             - monotonic seconds
      is_drowsy       - raw binary decision from the model
      face_detected   - bool
      warmup_active   - True only during initial warm-up
      pitch_dev_max   - max(|baseline - pitch|) over recent window, or None
      muted           - informational; arbiter returns the level regardless

    Priority (per spec):
      FACE_LOST > warmup->IDLE > INTENSE > MILD > IDLE
    """

    def __init__(self,
                 enter_sec: float = 3.0,
                 intense_onset_sec: float = 12.0,
                 release_sec: float = 3.0,
                 face_lost_delay_sec: float = 2.0,
                 face_lost_release_sec: float = 5.0,
                 pitch_severe_deg: float = 35.0):
        self.enter_sec           = float(enter_sec)
        self.intense_onset_sec   = float(intense_onset_sec)
        self.release_sec         = float(release_sec)
        self.face_lost_delay_sec = float(face_lost_delay_sec)
        self.face_lost_release_sec = float(face_lost_release_sec)
        self.pitch_severe_deg    = float(pitch_severe_deg)
        self.reset()

    def reset(self):
        self._drowsy_run_start: Optional[float] = None
        self._non_drowsy_run_start: Optional[float] = None
        self._mild_latched = False
        self._intense_latched = False

        self._face_lost_started: Optional[float] = None
        self._face_lost_latched = False
        self._face_return_non_drowsy_start: Optional[float] = None

        self._initial_warmup_done = False
        self._any_face_detected   = False

    @property
    def initial_warmup_done(self) -> bool:
        return self._initial_warmup_done

    def update(self, now: float, *,
               is_drowsy: bool,
               face_detected: bool,
               warmup_active: bool,
               pitch_dev_max: Optional[float],
               muted: bool = False) -> AlarmState:

        # --- Initial warm-up end ---
        if not self._initial_warmup_done and not warmup_active:
            self._initial_warmup_done = True
            if not self._any_face_detected:
                # Blanket case: no face the entire warm-up.
                self._face_lost_latched = True

        if face_detected:
            self._any_face_detected = True

        # --- Face-lost tracking ---
        if face_detected:
            self._face_lost_started = None
        else:
            if self._face_lost_started is None:
                self._face_lost_started = now
            if (not self._face_lost_latched
                and (now - self._face_lost_started) >= self.face_lost_delay_sec
                and pitch_dev_max is not None
                and pitch_dev_max >= self.pitch_severe_deg):
                self._face_lost_latched = True

        # --- Drowsy sub-FSM (frozen during warm-up or face loss) ---
        if self._initial_warmup_done and face_detected:
            if is_drowsy:
                if self._drowsy_run_start is None:
                    self._drowsy_run_start = now
                self._non_drowsy_run_start = None

                run = now - self._drowsy_run_start
                if run >= self.enter_sec:
                    self._mild_latched = True
                if run >= self.intense_onset_sec:
                    self._intense_latched = True
            else:
                if self._non_drowsy_run_start is None:
                    self._non_drowsy_run_start = now
                self._drowsy_run_start = None  # breaks 12 s continuity

                if (now - self._non_drowsy_run_start) >= self.release_sec:
                    self._mild_latched = False
                    self._intense_latched = False

        # --- Face-lost release (5 s continuous non-drowsy) ---
        if self._face_lost_latched and face_detected:
            if is_drowsy:
                self._face_return_non_drowsy_start = None
            else:
                if self._face_return_non_drowsy_start is None:
                    self._face_return_non_drowsy_start = now
                if (now - self._face_return_non_drowsy_start) >= self.face_lost_release_sec:
                    self._face_lost_latched = False
                    self._face_return_non_drowsy_start = None
        elif not face_detected:
            self._face_return_non_drowsy_start = None

        # --- Arbiter ---
        if self._face_lost_latched:
            level = AlarmLevel.FACE_LOST
        elif not self._initial_warmup_done:
            level = AlarmLevel.IDLE
        elif self._intense_latched:
            level = AlarmLevel.INTENSE
        elif self._mild_latched:
            level = AlarmLevel.MILD
        else:
            level = AlarmLevel.IDLE

        drowsy_run_s = (now - self._drowsy_run_start
                        if self._drowsy_run_start is not None else 0.0)
        non_drowsy_run_s = (now - self._non_drowsy_run_start
                            if self._non_drowsy_run_start is not None else 0.0)
        face_lost_sec = (now - self._face_lost_started
                         if self._face_lost_started is not None else 0.0)

        return AlarmState(
            level=level,
            drowsy_run_s=drowsy_run_s,
            non_drowsy_run_s=non_drowsy_run_s,
            face_lost_sec=face_lost_sec,
            face_lost_latched=self._face_lost_latched,
            mild_latched=self._mild_latched,
            intense_latched=self._intense_latched,
            initial_warmup_done=self._initial_warmup_done,
        )