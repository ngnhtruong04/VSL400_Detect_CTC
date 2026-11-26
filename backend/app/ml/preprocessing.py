"""Strict Parity Preprocessing pipeline for Vietnamese Sign Language (VSL400) hands-only representation.

Hard Parity requirements:
- Body-centric shoulder normalization:
  * center = (left_shoulder + right_shoulder) / 2
  * scale = max(||left_shoulder - right_shoulder||_2 * multiplier, min_scale)
  * coords = clip((coords - center) / scale, -clip_value, clip_value)
- 42 hand joints:
  * 0–20: Left Hand (21 joints)
  * 21–41: Right Hand (21 joints)
- 7 features per joint: [x, y, z, dx, dy, dz, valid]
- Temporal delta:
  * dxyz[0] = 0.0
  * dxyz[t] = coords[t] - coords[t-1] ONLY when joint was valid in both t and t-1, else 0.0
- Missing landmark contract:
  * coords = 0.0
  * valid = 0.0
  * dxyz = 0.0
  * NO fake landmark injection or persistence hold
- Flattened output: (T, 294) with dtype float32
"""

from collections import deque
from typing import List, Optional, Tuple, Dict, Any
import numpy as np


class VSLPreprocessor:
    def __init__(
        self,
        pose_visibility_threshold: float = 0.4,
        pose_presence_threshold: float = 0.2,
        min_scale: float = 0.0001,
        shoulder_scale_multiplier: float = 1.0,
        clip_value: float = 6.0,
        n_joints: int = 42,
        features_per_joint: int = 7,
    ):
        self.pose_visibility_threshold = pose_visibility_threshold
        self.pose_presence_threshold = pose_presence_threshold
        self.min_scale = min_scale
        self.shoulder_scale_multiplier = shoulder_scale_multiplier
        self.clip_value = clip_value
        self.n_joints = n_joints
        self.features_per_joint = features_per_joint
        self.flattened_dim = n_joints * features_per_joint  # 42 * 7 = 294

        # Temporal anchor and velocity state (3D model features — strictly unchanged)
        self.last_center: Optional[np.ndarray] = None
        self.last_scale: Optional[float] = None
        self.prev_hand_coords: Optional[np.ndarray] = None
        self.prev_hand_valid: Optional[np.ndarray] = None

        # Dedicated 2D smoothed-displacement history for FSM/Auto-Stop motion_energy ONLY
        # (never modifies the 294-dim 3D feature vector fed to HandsTCNTransformerCTC)
        self._motion_raw_xy_hist: deque = deque(maxlen=3)
        self._motion_smooth_xy_hist: deque = deque(maxlen=6)

        # Signing space flag
        self.in_signing_space: bool = False

        # Diagnostic metadata for latest processed frame
        self.last_left_valid_ratio: float = 0.0
        self.last_right_valid_ratio: float = 0.0
        self.last_overall_valid_ratio: float = 0.0
        self.last_nan_count: int = 0
        self.last_inf_count: int = 0

    def reset_motion_state(self):
        """Reset only the FSM/Auto-Stop 2D motion history (keeps shoulder anchor)."""
        self._motion_raw_xy_hist.clear()
        self._motion_smooth_xy_hist.clear()

    def reset_state(self):
        """Reset temporal state between independent signing sessions."""
        self.last_center = None
        self.last_scale = None
        self.prev_hand_coords = None
        self.prev_hand_valid = None
        self.reset_motion_state()
        self.in_signing_space = False
        self.last_left_valid_ratio = 0.0
        self.last_right_valid_ratio = 0.0
        self.last_overall_valid_ratio = 0.0
        self.last_nan_count = 0
        self.last_inf_count = 0

    def compute_shoulder_anchor(
        self,
        pose_landmarks: Optional[List[Dict[str, float]]],
        frame_aspect: float = 1.0,
    ) -> Tuple[np.ndarray, float]:
        """Calculates shoulder center and normalization scale from Pose landmarks.
        Pose landmark 11: left shoulder
        Pose landmark 12: right shoulder
        Args:
            frame_aspect: H / W of source frame. Multiplying normalized y by H/W restores
                isotropic (x, y, z) geometry matching 1:1 square VSL400 training frames
                (exact identity when frame_aspect == 1.0).
        """
        aspect = float(frame_aspect) if (frame_aspect and np.isfinite(frame_aspect) and frame_aspect > 0.0) else 1.0
        default_center = np.array([0.5, 0.5 * aspect, 0.0], dtype=np.float32)
        default_scale = 1.0

        if not pose_landmarks or len(pose_landmarks) < 13:
            if self.last_center is not None and self.last_scale is not None:
                return self.last_center, self.last_scale
            return default_center, default_scale

        left_sh = pose_landmarks[11]
        right_sh = pose_landmarks[12]

        left_vis = left_sh.get("visibility", 1.0)
        right_vis = right_sh.get("visibility", 1.0)
        left_pres = left_sh.get("presence", 1.0)
        right_pres = right_sh.get("presence", 1.0)

        # Check if both shoulders satisfy detection confidence thresholds
        shoulders_valid = (
            left_vis >= self.pose_visibility_threshold
            and right_vis >= self.pose_visibility_threshold
            and left_pres >= self.pose_presence_threshold
            and right_pres >= self.pose_presence_threshold
        )

        if shoulders_valid:
            left_pt = np.array([left_sh["x"], left_sh["y"] * aspect], dtype=np.float32)
            right_pt = np.array([right_sh["x"], right_sh["y"] * aspect], dtype=np.float32)
            if np.all(np.isfinite(left_pt)) and np.all(np.isfinite(right_pt)):
                # 2D shoulder midpoint for (x, y_iso); keep center_z = 0.0 (exact match with NB00 training formula:
                # x'=(x-cx)/s, y'=(y_iso-cy_iso)/s, z'=z/s).
                mid_xy = (left_pt + right_pt) / 2.0
                center = np.array([mid_xy[0], mid_xy[1], 0.0], dtype=np.float32)
                dist = float(np.linalg.norm(left_pt - right_pt))
                scale = max(dist * self.shoulder_scale_multiplier, self.min_scale)

                self.last_center = center
                self.last_scale = scale
                return center, scale

        # Fallback 1: temporal anchor inheritance if shoulders briefly occluded
        if self.last_center is not None and self.last_scale is not None:
            return self.last_center, self.last_scale

        # Fallback 2: estimate from other valid upper-body pose points (exact NB00 fallback)
        valid_body_xy = []
        for idx in (0, 11, 12, 13, 14, 15, 16, 23, 24):
            if idx < len(pose_landmarks):
                pt = pose_landmarks[idx]
                vis = pt.get("visibility", 1.0)
                pres = pt.get("presence", 1.0)
                if vis >= self.pose_visibility_threshold and pres >= self.pose_presence_threshold:
                    xy = np.array([pt["x"], pt["y"] * aspect], dtype=np.float32)
                    if np.all(np.isfinite(xy)):
                        valid_body_xy.append(xy)
        if len(valid_body_xy) >= 2:
            xy_arr = np.stack(valid_body_xy, axis=0)
            mid_xy = xy_arr.mean(axis=0)
            scale = max(float(np.max(np.ptp(xy_arr, axis=0))) * self.shoulder_scale_multiplier, self.min_scale)
            return np.array([mid_xy[0], mid_xy[1], 0.0], dtype=np.float32), scale

        return default_center, default_scale

    def check_signing_space(
        self,
        left_hand_lms: Optional[List[Dict[str, float]]],
        right_hand_lms: Optional[List[Dict[str, float]]],
        pose_landmarks: Optional[List[Dict[str, float]]],
        frame_aspect: float = 1.0,
        motion_energy: float = 0.0,
        center: Optional[np.ndarray] = None,
        scale: Optional[float] = None,
    ) -> bool:
        """Determines if either hand is in the body-proportional active signing space.

        Uses isotropic shoulder-normalized vertical offset `rel_y = (y_iso - cy_iso) / scale`
        instead of a fixed image-height constant so near/far camera distances behave identically:
        - Enter signing space: `rel_y <= 1.25` shoulder widths below shoulder midpoint.
        - Exit signing space (hysteresis): `rel_y > 1.35` to prevent flicker on low signs/holds.
        - Motion-assisted lower-torso allowance: intentional directed motion (`motion_energy >= 0.025`)
          in the lower-torso band (`rel_y <= 1.50`) is NOT blindly rejected by spatial cutoff alone.
        - Falls back to legacy `shoulder_y + 0.35` only when no valid shoulder scale is available.
        """
        aspect = float(frame_aspect) if (frame_aspect and np.isfinite(frame_aspect) and frame_aspect > 0.0) else 1.0

        # Resolve body-proportional shoulder anchor (cy_iso, sh_scale)
        cy_iso: Optional[float] = None
        sh_scale: Optional[float] = None
        if center is not None and scale is not None and scale > 0.02 and np.isfinite(scale):
            # Only treat anchor as real body scale if shoulders were actually observed (not default scale=1.0)
            if self.last_scale is not None and abs(scale - self.last_scale) < 1e-4:
                cy_iso = float(center[1])
                sh_scale = float(scale)
        if cy_iso is None and pose_landmarks and len(pose_landmarks) > 12:
            left_sh = pose_landmarks[11]
            right_sh = pose_landmarks[12]
            if (
                left_sh.get("visibility", 1.0) >= self.pose_visibility_threshold
                and right_sh.get("visibility", 1.0) >= self.pose_visibility_threshold
                and left_sh.get("presence", 1.0) >= self.pose_presence_threshold
                and right_sh.get("presence", 1.0) >= self.pose_presence_threshold
            ):
                lp = np.array([left_sh["x"], left_sh["y"] * aspect], dtype=np.float32)
                rp = np.array([right_sh["x"], right_sh["y"] * aspect], dtype=np.float32)
                if np.all(np.isfinite(lp)) and np.all(np.isfinite(rp)):
                    dist = float(np.linalg.norm(lp - rp)) * self.shoulder_scale_multiplier
                    if dist > 0.02:
                        cy_iso = float((lp[1] + rp[1]) / 2.0)
                        sh_scale = dist
        if cy_iso is None and self.last_center is not None and self.last_scale is not None and self.last_scale > 0.02:
            cy_iso = float(self.last_center[1])
            sh_scale = float(self.last_scale)

        hand_min_y_raw: Optional[float] = None
        for hand_lms in (left_hand_lms, right_hand_lms):
            if hand_lms and len(hand_lms) > 0:
                wrist_y = hand_lms[0]["y"]
                tip_y = hand_lms[min(12, len(hand_lms) - 1)]["y"]
                cand = min(wrist_y, tip_y)
                if np.isfinite(cand):
                    hand_min_y_raw = cand if hand_min_y_raw is None else min(hand_min_y_raw, cand)

        if hand_min_y_raw is None:
            return False

        if cy_iso is not None and sh_scale is not None and sh_scale > 0.02:
            rel_y = (hand_min_y_raw * aspect - cy_iso) / sh_scale
            threshold = 1.35 if self.in_signing_space else 1.25
            if rel_y <= threshold:
                return True
            # Do not blindly reject low-torso signs when strong intentional directed motion is present
            if rel_y <= 1.50 and motion_energy >= 0.025:
                return True
            return False

        # Legacy fallback when no valid shoulder scale exists
        shoulder_y = 0.4
        if pose_landmarks and len(pose_landmarks) > 12:
            shoulder_y = (pose_landmarks[11].get("y", 0.4) + pose_landmarks[12].get("y", 0.4)) / 2.0
        return hand_min_y_raw <= (shoulder_y + 0.35)

    def normalize_hand_landmarks(
        self,
        left_hand_lms: Optional[List[Dict[str, float]]],
        right_hand_lms: Optional[List[Dict[str, float]]],
        center: np.ndarray,
        scale: float,
        frame_aspect: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Normalizes left and right hand coordinates (21 joints each) relative to shoulder anchor.
        Strict Contract:
        - If hand missing: coords = 0.0, valid = 0.0
        - If hand present: (coords - center) / scale clipped to [-clip_value, clip_value], valid = 1.0
        Returns:
            coords: np.ndarray of shape (42, 3)
            valid: np.ndarray of shape (42,)
        """
        aspect = float(frame_aspect) if (frame_aspect and np.isfinite(frame_aspect) and frame_aspect > 0.0) else 1.0
        coords = np.zeros((42, 3), dtype=np.float32)
        valid = np.zeros(42, dtype=np.float32)

        # Left hand: indices 0 to 20
        if left_hand_lms and len(left_hand_lms) == 21:
            for i, lm in enumerate(left_hand_lms):
                raw_pt = np.array([lm["x"], lm["y"] * aspect, lm.get("z", 0.0)], dtype=np.float32)
                if np.all(np.isfinite(raw_pt)):
                    norm_pt = (raw_pt - center) / scale
                    coords[i] = np.clip(norm_pt, -self.clip_value, self.clip_value)
                    valid[i] = 1.0
                else:
                    coords[i] = 0.0
                    valid[i] = 0.0

        # Right hand: indices 21 to 41
        if right_hand_lms and len(right_hand_lms) == 21:
            for i, lm in enumerate(right_hand_lms):
                raw_pt = np.array([lm["x"], lm["y"] * aspect, lm.get("z", 0.0)], dtype=np.float32)
                if np.all(np.isfinite(raw_pt)):
                    norm_pt = (raw_pt - center) / scale
                    coords[21 + i] = np.clip(norm_pt, -self.clip_value, self.clip_value)
                    valid[21 + i] = 1.0
                else:
                    coords[21 + i] = 0.0
                    valid[21 + i] = 0.0

        return coords, valid

    def _compute_fsm_motion_energy(self, coords: np.ndarray, valid: np.ndarray) -> float:
        """Computes jitter-resistant 2D net hand motion energy for FSM / Auto-Stop ONLY.

        Uses a 3-frame validity-aware moving average of normalized 2D (x, y) coordinates
        followed by span-5 net displacement per frame. This cancels MediaPipe Hands
        period-2/3 ROI limit-cycle oscillation and excludes noisy monocular depth (dz),
        while preserving low-frequency directed signing motion (including slow 0.5x signs).
        Does NOT alter the 3D [x, y, z, dx, dy, dz, valid] tensor fed to the model.
        """
        xy = coords[:, :2].copy()
        valid_bool = valid == 1.0
        self._motion_raw_xy_hist.append((xy, valid_bool))

        acc = np.zeros_like(xy, dtype=np.float32)
        cnt = np.zeros(self.n_joints, dtype=np.float32)
        for h_xy, h_v in self._motion_raw_xy_hist:
            acc += np.where(h_v[:, np.newaxis], h_xy, 0.0)
            cnt += h_v.astype(np.float32)
        smooth_xy = np.where(
            (cnt > 0.0)[:, np.newaxis],
            acc / np.maximum(cnt[:, np.newaxis], 1.0),
            0.0,
        ).astype(np.float32)

        self._motion_smooth_xy_hist.append((smooth_xy, valid_bool))
        if len(self._motion_smooth_xy_hist) < 2:
            return 0.0

        xy_start, v_start = self._motion_smooth_xy_hist[0]
        xy_end, v_end = self._motion_smooth_xy_hist[-1]
        both_valid = v_start & v_end
        if not np.any(both_valid):
            return 0.0

        span = len(self._motion_smooth_xy_hist) - 1
        disp = np.linalg.norm(xy_end[both_valid] - xy_start[both_valid], axis=-1)
        return float(np.mean(disp) / span)

    def process_frame(
        self,
        left_hand_lms: Optional[List[Dict[str, float]]],
        right_hand_lms: Optional[List[Dict[str, float]]],
        pose_lms: Optional[List[Dict[str, float]]],
        frame_aspect: float = 1.0,
    ) -> Tuple[np.ndarray, float]:
        """Processes a single frame into a flattened 294-dim feature vector strictly matching training parity.
        Returns:
            frame_features: np.ndarray of shape (294,)
            motion_energy: float representing jitter-resistant 2D hand motion energy for FSM
        """
        center, scale = self.compute_shoulder_anchor(pose_lms, frame_aspect=frame_aspect)
        coords, valid = self.normalize_hand_landmarks(
            left_hand_lms, right_hand_lms, center, scale, frame_aspect=frame_aspect
        )

        # Track valid ratios
        self.last_left_valid_ratio = float(np.mean(valid[:21]))
        self.last_right_valid_ratio = float(np.mean(valid[21:]))
        self.last_overall_valid_ratio = float(np.mean(valid))

        # Compute temporal delta dxyz (strict 3D model feature contract — unchanged!)
        dxyz = np.zeros((42, 3), dtype=np.float32)

        if self.prev_hand_coords is not None and self.prev_hand_valid is not None:
            # Joint is only valid for velocity if both current and previous frames were detected
            joint_valid_mask = (valid == 1.0) & (self.prev_hand_valid == 1.0)
            dxyz[joint_valid_mask] = coords[joint_valid_mask] - self.prev_hand_coords[joint_valid_mask]

        # Compute decoupled 2D smoothed net motion energy strictly for FSM / Auto-Stop
        motion_energy = self._compute_fsm_motion_energy(coords, valid)

        # Body-proportional scale-invariant signing space check (with hysteresis + motion allowance)
        self.in_signing_space = self.check_signing_space(
            left_hand_lms,
            right_hand_lms,
            pose_lms,
            frame_aspect=frame_aspect,
            motion_energy=motion_energy,
            center=center,
            scale=scale,
        )

        # Construct 7 features per joint: [x, y, z, dx, dy, dz, valid] -> shape (42, 7)
        features_42x7 = np.concatenate([
            coords,                 # (42, 3)
            dxyz,                   # (42, 3)
            valid[:, np.newaxis],   # (42, 1)
        ], axis=-1).astype(np.float32)

        # Flatten to (294,)
        flat_features = features_42x7.reshape(-1)

        # Check for NaN / Inf
        self.last_nan_count = int(np.isnan(flat_features).sum())
        self.last_inf_count = int(np.isinf(flat_features).sum())
        if self.last_nan_count > 0 or self.last_inf_count > 0:
            flat_features = np.nan_to_num(flat_features, nan=0.0, posinf=self.clip_value, neginf=-self.clip_value)

        # Update temporal state
        self.prev_hand_coords = coords.copy()
        self.prev_hand_valid = valid.copy()

        return flat_features, motion_energy
