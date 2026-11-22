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

        # Temporal anchor and velocity state
        self.last_center: Optional[np.ndarray] = None
        self.last_scale: Optional[float] = None
        self.prev_hand_coords: Optional[np.ndarray] = None
        self.prev_hand_valid: Optional[np.ndarray] = None

        # Signing space flag
        self.in_signing_space: bool = False

        # Diagnostic metadata for latest processed frame
        self.last_left_valid_ratio: float = 0.0
        self.last_right_valid_ratio: float = 0.0
        self.last_overall_valid_ratio: float = 0.0
        self.last_nan_count: int = 0
        self.last_inf_count: int = 0

    def reset_state(self):
        """Reset temporal state between independent signing sessions."""
        self.last_center = None
        self.last_scale = None
        self.prev_hand_coords = None
        self.prev_hand_valid = None
        self.in_signing_space = False
        self.last_left_valid_ratio = 0.0
        self.last_right_valid_ratio = 0.0
        self.last_overall_valid_ratio = 0.0
        self.last_nan_count = 0
        self.last_inf_count = 0

    def compute_shoulder_anchor(
        self,
        pose_landmarks: Optional[List[Dict[str, float]]],
    ) -> Tuple[np.ndarray, float]:
        """Calculates shoulder center and normalization scale from Pose landmarks.
        Pose landmark 11: left shoulder
        Pose landmark 12: right shoulder
        """
        default_center = np.array([0.5, 0.5, 0.0], dtype=np.float32)
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
        if (
            left_vis < self.pose_visibility_threshold
            or right_vis < self.pose_visibility_threshold
            or left_pres < self.pose_presence_threshold
            or right_pres < self.pose_presence_threshold
        ):
            if self.last_center is not None and self.last_scale is not None:
                return self.last_center, self.last_scale
            return default_center, default_scale

        left_pt = np.array([left_sh["x"], left_sh["y"]], dtype=np.float32)
        right_pt = np.array([right_sh["x"], right_sh["y"]], dtype=np.float32)

        if not np.all(np.isfinite(left_pt)) or not np.all(np.isfinite(right_pt)):
            if self.last_center is not None and self.last_scale is not None:
                return self.last_center, self.last_scale
            return default_center, default_scale

        # 2D shoulder midpoint for (x, y); keep center_z = 0.0 because MediaPipe Hands z is
        # already wrist-relative (wrist.z == 0) whereas Pose shoulder.z is hip-centered (~ -0.5).
        # Subtracting Pose shoulder.z from Hand z creates a bogus +2..+3 Z-shift & noisy wrist dz.
        mid_xy = (left_pt + right_pt) / 2.0
        center = np.array([mid_xy[0], mid_xy[1], 0.0], dtype=np.float32)
        dist = float(np.linalg.norm(left_pt - right_pt))  # 2D Euclidean distance in image plane
        scale = max(dist * self.shoulder_scale_multiplier, self.min_scale)

        self.last_center = center
        self.last_scale = scale
        return center, scale

    def check_signing_space(
        self,
        left_hand_lms: Optional[List[Dict[str, float]]],
        right_hand_lms: Optional[List[Dict[str, float]]],
        pose_landmarks: Optional[List[Dict[str, float]]],
    ) -> bool:
        """Determines if either hand is elevated into the active signing space.
        Resting zone: hands down at hips/lap (y > shoulder_y + 0.35).
        Signing space: hands elevated in front of torso, chest, or head (y <= shoulder_y + 0.35).
        """
        shoulder_y = 0.4
        if pose_landmarks and len(pose_landmarks) > 12:
            left_sh_y = pose_landmarks[11].get("y", 0.4)
            right_sh_y = pose_landmarks[12].get("y", 0.4)
            shoulder_y = (left_sh_y + right_sh_y) / 2.0

        max_y_threshold = shoulder_y + 0.35

        if left_hand_lms and len(left_hand_lms) > 0:
            wrist_y = left_hand_lms[0]["y"]
            tip_y = left_hand_lms[min(12, len(left_hand_lms) - 1)]["y"]
            if min(wrist_y, tip_y) <= max_y_threshold:
                return True

        if right_hand_lms and len(right_hand_lms) > 0:
            wrist_y = right_hand_lms[0]["y"]
            tip_y = right_hand_lms[min(12, len(right_hand_lms) - 1)]["y"]
            if min(wrist_y, tip_y) <= max_y_threshold:
                return True

        return False

    def normalize_hand_landmarks(
        self,
        left_hand_lms: Optional[List[Dict[str, float]]],
        right_hand_lms: Optional[List[Dict[str, float]]],
        center: np.ndarray,
        scale: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Normalizes left and right hand coordinates (21 joints each) relative to shoulder anchor.
        Strict Contract:
        - If hand missing: coords = 0.0, valid = 0.0
        - If hand present: (coords - center) / scale clipped to [-clip_value, clip_value], valid = 1.0
        Returns:
            coords: np.ndarray of shape (42, 3)
            valid: np.ndarray of shape (42,)
        """
        coords = np.zeros((42, 3), dtype=np.float32)
        valid = np.zeros(42, dtype=np.float32)

        # Left hand: indices 0 to 20
        if left_hand_lms and len(left_hand_lms) == 21:
            for i, lm in enumerate(left_hand_lms):
                raw_pt = np.array([lm["x"], lm["y"], lm.get("z", 0.0)], dtype=np.float32)
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
                raw_pt = np.array([lm["x"], lm["y"], lm.get("z", 0.0)], dtype=np.float32)
                if np.all(np.isfinite(raw_pt)):
                    norm_pt = (raw_pt - center) / scale
                    coords[21 + i] = np.clip(norm_pt, -self.clip_value, self.clip_value)
                    valid[21 + i] = 1.0
                else:
                    coords[21 + i] = 0.0
                    valid[21 + i] = 0.0

        return coords, valid

    def process_frame(
        self,
        left_hand_lms: Optional[List[Dict[str, float]]],
        right_hand_lms: Optional[List[Dict[str, float]]],
        pose_lms: Optional[List[Dict[str, float]]],
    ) -> Tuple[np.ndarray, float]:
        """Processes a single frame into a flattened 294-dim feature vector strictly matching training parity.
        Returns:
            frame_features: np.ndarray of shape (294,)
            motion_energy: float representing total hand motion energy for FSM
        """
        # Check signing space elevation
        self.in_signing_space = self.check_signing_space(left_hand_lms, right_hand_lms, pose_lms)

        center, scale = self.compute_shoulder_anchor(pose_lms)
        coords, valid = self.normalize_hand_landmarks(left_hand_lms, right_hand_lms, center, scale)

        # Track valid ratios
        self.last_left_valid_ratio = float(np.mean(valid[:21]))
        self.last_right_valid_ratio = float(np.mean(valid[21:]))
        self.last_overall_valid_ratio = float(np.mean(valid))

        # Compute temporal delta dxyz
        dxyz = np.zeros((42, 3), dtype=np.float32)
        motion_energy = 0.0

        if self.prev_hand_coords is not None and self.prev_hand_valid is not None:
            # Joint is only valid for velocity if both current and previous frames were detected
            joint_valid_mask = (valid == 1.0) & (self.prev_hand_valid == 1.0)
            dxyz[joint_valid_mask] = coords[joint_valid_mask] - self.prev_hand_coords[joint_valid_mask]
            if np.any(joint_valid_mask):
                motion_energy = float(np.mean(np.linalg.norm(dxyz[joint_valid_mask], axis=-1)))

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
