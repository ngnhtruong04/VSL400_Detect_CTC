"""Endpoint Finite State Machine (FSM) for Vietnamese Sign Language utterances.

States:
- IDLE: Waiting for sign activity
- STARTING: Motion detected in signing space, confirming active sign
- SIGNING: Active signing in progress, live gloss emitted
- POSSIBLE_END: Motion paused briefly, waiting to distinguish sign transition from sentence end
- END_OF_UTTERANCE: Utterance finalized, final gloss committed
- COOLDOWN: Short cooldown before resetting to IDLE
"""

import time
import math
from enum import Enum
from typing import List, Optional, Tuple


class FSMState(str, Enum):
    IDLE = "IDLE"
    STARTING = "STARTING"
    SIGNING = "SIGNING"
    POSSIBLE_END = "POSSIBLE_END"
    END_OF_UTTERANCE = "END_OF_UTTERANCE"
    COOLDOWN = "COOLDOWN"


class EndpointFSM:
    def __init__(
        self,
        motion_threshold: float = 0.015,
        min_sign_duration_ms: int = 400,
        possible_end_delay_ms: int = 600,
        silence_ms: int = 1200,
        cooldown_ms: int = 500,
        hands_lowered_possible_end_ms: int = 650,
        hands_lowered_commit_ms: int = 1200,
    ):
        self.motion_threshold = motion_threshold
        self.min_sign_duration_sec = min_sign_duration_ms / 1000.0
        self.possible_end_delay_sec = possible_end_delay_ms / 1000.0
        self.silence_sec = silence_ms / 1000.0
        self.cooldown_sec = cooldown_ms / 1000.0
        self.hands_lowered_possible_end_sec = hands_lowered_possible_end_ms / 1000.0
        self.hands_lowered_commit_sec = hands_lowered_commit_ms / 1000.0

        self.state: FSMState = FSMState.IDLE
        self.state_entered_time: float = time.time()
        self.signing_start_time: Optional[float] = None
        self.last_motion_time: float = time.time()
        self.last_update_time: Optional[float] = None
        self.smoothed_motion: Optional[float] = None

        # Utterance prediction tracking
        self.current_live_gloss: List[str] = []
        self.last_final_gloss: List[str] = []

    def reset(self):
        """Full reset to IDLE state."""
        self.state = FSMState.IDLE
        self.state_entered_time = time.time()
        self.signing_start_time = None
        self.last_motion_time = time.time()
        self.last_update_time = None
        self.smoothed_motion = None
        self.current_live_gloss = []

    def _set_state(self, new_state: FSMState, now: float):
        self.state = new_state
        self.state_entered_time = now

    def update(
        self,
        motion_energy: float,
        has_hands: bool,
        blank_ratio: float,
        live_gloss: List[str],
        now: Optional[float] = None,
        in_signing_space: bool = True,
    ) -> Tuple[FSMState, bool]:
        """Updates FSM state given current frame metrics.
        Returns:
            (state, is_final_commit)
        """
        if now is None:
            now = time.time()

        dt = max(now - self.last_update_time, 0.0) if self.last_update_time is not None else 0.033
        self.last_update_time = now

        # Time-aware EMA smoothing
        if self.smoothed_motion is None:
            self.smoothed_motion = motion_energy
        else:
            # Fast attack (alpha ~0.6 per 33ms), smooth decay (decay rate ~ 5.0 / sec)
            if motion_energy > self.smoothed_motion:
                self.smoothed_motion = 0.6 * motion_energy + 0.4 * self.smoothed_motion
            else:
                decay = math.exp(-dt * 5.0)
                self.smoothed_motion = motion_energy + decay * (self.smoothed_motion - motion_energy)

        # Active motion detection: true if current frame or smoothed motion meets threshold
        is_active_motion = (
            (motion_energy >= self.motion_threshold)
            and has_hands
            and in_signing_space
        )

        if is_active_motion:
            self.last_motion_time = now

        if len(live_gloss) > 0:
            self.current_live_gloss = live_gloss

        commit_final = False

        if self.state == FSMState.IDLE:
            if is_active_motion:
                self.signing_start_time = now
                self.last_motion_time = now
                self._set_state(FSMState.STARTING, now)

        elif self.state == FSMState.STARTING:
            if (now - self.signing_start_time) >= self.min_sign_duration_sec:
                self._set_state(FSMState.SIGNING, now)
            elif not in_signing_space and (now - self.last_motion_time > 0.6):
                self._set_state(FSMState.IDLE, now)
            elif (now - self.last_motion_time) > 1.0:
                self._set_state(FSMState.IDLE, now)

        elif self.state == FSMState.SIGNING:
            time_since_motion = now - self.last_motion_time
            if (
                time_since_motion >= self.possible_end_delay_sec
                or (not in_signing_space and time_since_motion >= self.hands_lowered_possible_end_sec)
            ):
                self._set_state(FSMState.POSSIBLE_END, now)

        elif self.state == FSMState.POSSIBLE_END:
            # If motion resumes during pause window, seamlessly return to SIGNING
            if is_active_motion:
                self.last_motion_time = now
                self._set_state(FSMState.SIGNING, now)
            else:
                time_since_motion = now - self.last_motion_time
                # Utterance complete if silent for silence_sec OR hands clearly lowered past commit threshold
                if (
                    time_since_motion >= self.silence_sec
                    or (not in_signing_space and time_since_motion >= self.hands_lowered_commit_sec)
                ):
                    self.last_final_gloss = list(self.current_live_gloss)
                    commit_final = True
                    self._set_state(FSMState.END_OF_UTTERANCE, now)

        elif self.state == FSMState.END_OF_UTTERANCE:
            # Transition to cooldown
            self._set_state(FSMState.COOLDOWN, now)

        elif self.state == FSMState.COOLDOWN:
            if (now - self.state_entered_time) >= self.cooldown_sec:
                self.reset()

        return self.state, commit_final
