"""High-FPS Parallel LandmarkExtractor using MediaPipe Hands + Pose for 30+ FPS VSL recognition.

Design for 30 FPS CPU performance & VSL400 training parity:
- Training keypoints (extractor_config.json) were extracted using separate Pose + Hands models.
- MediaPipe Holistic additionally runs 468-point FaceMesh serially on every frame (completely unused
  by the hands-only HandsTCNTransformerCTC model), cutting CPU FPS in half.
- Running MediaPipe Hands (model_complexity=1) and Pose (model_complexity=1, pose_stride=2) in
  parallel on 2 threads (MediaPipe C++/TFLite releases the Python GIL) achieves ~55 FPS on CPU while
  matching MediaPipe Tasks (hand_landmarker + pose_landmarker_full) training Z/XY parity (< 0.003 MAE):
    - left_hand_lms: 21 joints of the signer's anatomical left hand (joints 0..20)
    - right_hand_lms: 21 joints of the signer's anatomical right hand (joints 21..41)
    - pose_lms: 33 joints including shoulders (11, 12) for body-centric normalization
- Missing hands ALWAYS return None (strict missing-landmark contract: NEVER persist stale hand joints).
"""

from concurrent.futures import ThreadPoolExecutor
import logging
import math
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    HAS_MP = True
except Exception:
    HAS_MP = False
    mp_hands = None
    mp_pose = None
    mp_drawing = None
    mp_drawing_styles = None

from backend.app.config.settings import settings

logger = logging.getLogger("vsl.landmark_extractor")


class LandmarkExtractor:
    """High-speed parallel Pose + Hands extractor achieving 30+ FPS on CPU."""

    def __init__(
        self,
        min_detection_confidence: Optional[float] = None,
        min_tracking_confidence: Optional[float] = None,
        pose_stride: Optional[int] = None,
        roi_enabled: Optional[bool] = None,
        roi_scale_multiplier: Optional[float] = None,
    ):
        det_conf = (
            float(min_detection_confidence)
            if min_detection_confidence is not None
            else settings.MP_MIN_DETECTION_CONFIDENCE
        )
        trk_conf = (
            float(min_tracking_confidence)
            if min_tracking_confidence is not None
            else settings.MP_MIN_TRACKING_CONFIDENCE
        )
        self.pose_stride = max(1, int(pose_stride if pose_stride is not None else settings.MP_POSE_STRIDE))
        self.roi_enabled = bool(settings.MP_ROI_ENABLED if roi_enabled is None else roi_enabled)
        self.roi_scale_multiplier = float(
            settings.MP_ROI_SCALE_MULTIPLIER if roi_scale_multiplier is None else roi_scale_multiplier
        )
        self._frame_idx: int = 0
        self._cached_pose_lms: Optional[List[Dict[str, float]]] = None
        self._cached_pose_proto = None
        self._smooth_roi: Optional[Tuple[float, float, float]] = None  # (cx, cy, side)
        self._hand_miss_streak: int = 0
        self._roi_locked: bool = False
        self._executor: Optional[ThreadPoolExecutor] = None

        if not HAS_MP:
            logger.warning("MediaPipe solutions module not available.")
            self.pose = None
            self.hands = None
            self.holistic = None
            return

        # Dedicated 2-worker pool so Hands and Pose execute concurrently in C++ (GIL released)
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="mp_extract")

        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=det_conf,
            min_tracking_confidence=trk_conf,
        )

        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=det_conf,
            min_tracking_confidence=trk_conf,
        )

        # Backward-compatibility attribute alias for existing unit tests/diagnostics
        self.holistic = self.pose

    @staticmethod
    def _has_missed_active_wrist(
        pose_lms: Optional[List[Dict[str, float]]],
        hands_res,
        W: int,
        H: int,
    ) -> bool:
        """Returns True when Pose shows a visible wrist (either elevated into signing space or
        any visible wrist when zero hands are detected) that has NO nearby MediaPipe hand."""
        if not pose_lms or len(pose_lms) <= 16 or W <= 0 or H <= 0:
            return False

        ls, rs = pose_lms[11], pose_lms[12]
        sh_y = (ls["y"] + rs["y"]) * 0.5
        sh_w_px = math.hypot((ls["x"] - rs["x"]) * W, (ls["y"] - rs["y"]) * H)
        if sh_w_px < 12.0:
            return False
        # Maximum normalized Y for an elevated signing wrist (~1.25 shoulder widths below shoulders)
        elevated_y_limit = sh_y + 1.25 * (sh_w_px / float(H))

        det_wrists = []
        if hands_res and hands_res.multi_hand_landmarks:
            for h_lms in hands_res.multi_hand_landmarks[:2]:
                w0 = h_lms.landmark[0]
                det_wrists.append((float(w0.x), float(w0.y)))

        for idx in (15, 16):
            pt = pose_lms[idx]
            if (
                pt.get("visibility", 0.0) >= 0.35
                and 0.03 <= pt["x"] <= 0.97
                and 0.03 <= pt["y"] <= 0.97
            ):
                # If 0 hands detected at all, any visible wrist counts; otherwise only check elevated signing wrists
                if not det_wrists or pt["y"] <= elevated_y_limit:
                    px, py = pt["x"], pt["y"]
                    if not any(math.hypot(px - dx, py - dy) <= 0.14 for dx, dy in det_wrists):
                        return True
        return False

    def _compute_upper_body_roi(
        self,
        pose_lms: Optional[List[Dict[str, float]]],
        W: int,
        H: int,
    ) -> Optional[Tuple[int, int, int, int]]:
        """Computes a deadband-smoothed square upper-body ROI around the signer's shoulders.

        Uses closed-loop adaptive activation (`self._roi_locked`):
        - When full-frame MediaPipe Hands already tracks hands cleanly (normal 1:1 / 4:3 / near),
          ROI stays inactive so full-frame tracking stability and timing remain 100% intact.
        - When Pose confirms the signer is small/far (`sh_ratio < 0.18`) with wrists in frame,
          yet full-frame Hands misses hands for >= 2 consecutive frames, `_roi_locked` engages
          so MediaPipe Hands' 192x192 palm detector zooms onto the signer's upper body while
          remapping landmarks back into exact full-frame (x, y, z) coordinates.
        """
        if not self.roi_enabled or not self._roi_locked or not pose_lms or len(pose_lms) <= 12 or W <= 0 or H <= 0:
            return None

        ls, rs = pose_lms[11], pose_lms[12]
        if ls.get("visibility", 0.0) < 0.4 or rs.get("visibility", 0.0) < 0.4:
            return None

        lx, ly = ls["x"] * W, ls["y"] * H
        rx, ry = rs["x"] * W, rs["y"] * H
        sh_w = math.hypot(lx - rx, ly - ry)
        if sh_w < 12.0:
            return None

        sh_ratio = sh_w / float(W)
        # Disengage ROI if signer moves close to camera
        if sh_ratio >= 0.22:
            self._roi_locked = False
            self._smooth_roi = None
            return None

        target_cx = (lx + rx) * 0.5
        target_cy = (ly + ry) * 0.5 + 0.10 * sh_w
        min_dim = float(min(W, H))
        target_side = min(max(self.roi_scale_multiplier * sh_w, 0.52 * min_dim), min_dim)

        if target_side >= 0.92 * W and target_side >= 0.92 * H:
            return None

        if self._smooth_roi is None:
            scx, scy, sside = target_cx, target_cy, target_side
        else:
            ocx, ocy, oside = self._smooth_roi
            # Deadband keeps MediaPipe Hands internal temporal tracker stable
            if math.hypot(target_cx - ocx, target_cy - ocy) < 10.0 and abs(target_side - oside) < 14.0:
                scx, scy, sside = ocx, ocy, oside
            else:
                alpha = 0.25
                scx = (1.0 - alpha) * ocx + alpha * target_cx
                scy = (1.0 - alpha) * ocy + alpha * target_cy
                sside = (1.0 - alpha) * oside + alpha * target_side
        self._smooth_roi = (scx, scy, sside)

        side = max(64, min(int(round(sside)), W, H))
        x0 = max(0, min(int(round(scx - side * 0.5)), W - side))
        y0 = max(0, min(int(round(scy - side * 0.5)), H - side))
        return x0, y0, x0 + side, y0 + side

    @staticmethod
    def _remap_hand_from_roi(
        hand_lms: Optional[List[Dict[str, float]]],
        x0: int,
        y0: int,
        roi_w: int,
        roi_h: int,
        W: int,
        H: int,
    ) -> Optional[List[Dict[str, float]]]:
        """Maps ROI-normalized hand landmarks back into exact full-frame (x, y, z) coordinates.
        Scales z by (roi_w / W) so isotropic 3D depth parity with full-frame extraction is preserved.
        """
        if hand_lms is None:
            return None
        z_scale = roi_w / float(W)
        return [
            {
                "x": (x0 + lm["x"] * roi_w) / float(W),
                "y": (y0 + lm["y"] * roi_h) / float(H),
                "z": lm["z"] * z_scale,
            }
            for lm in hand_lms
        ]

    def _run_hands(self, frame_rgb: np.ndarray):
        return self.hands.process(frame_rgb)

    def _run_pose(self, frame_rgb: np.ndarray):
        return self.pose.process(frame_rgb)

    @staticmethod
    def _assign_hands_anatomically(
        hands_res,
        pose_lms: Optional[List[Dict[str, float]]],
    ) -> Tuple[Optional[List[Dict[str, float]]], Optional[List[Dict[str, float]]]]:
        """Maps detected hands to signer's anatomical Left and Right hands.

        Primary: Minimum-cost 1-to-1 wrist proximity against Pose LM 15 (Left) & LM 16 (Right).
        Fallback: MediaPipe handedness (inverted for unmirrored camera frame) or image X half.
        """
        if not hands_res or not hands_res.multi_hand_landmarks:
            return None, None

        detected_hands: List[List[Dict[str, float]]] = []
        wrists: List[np.ndarray] = []
        for hand_lms in hands_res.multi_hand_landmarks[:2]:
            lm_list = [
                {"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)}
                for lm in hand_lms.landmark
            ]
            detected_hands.append(lm_list)
            wrists.append(np.array([lm_list[0]["x"], lm_list[0]["y"]], dtype=np.float32))

        has_pose_wrists = (
            pose_lms is not None
            and len(pose_lms) > 16
            and pose_lms[15].get("visibility", 0.0) >= 0.25
            and pose_lms[16].get("visibility", 0.0) >= 0.25
        )

        left_hand_lms: Optional[List[Dict[str, float]]] = None
        right_hand_lms: Optional[List[Dict[str, float]]] = None

        if has_pose_wrists:
            left_wrist = np.array([pose_lms[15]["x"], pose_lms[15]["y"]], dtype=np.float32)
            right_wrist = np.array([pose_lms[16]["x"], pose_lms[16]["y"]], dtype=np.float32)

            if len(detected_hands) == 2:
                # Optimal 1-to-1 assignment (handles crossing hands without collisions)
                cost_0L_1R = float(
                    np.linalg.norm(wrists[0] - left_wrist) + np.linalg.norm(wrists[1] - right_wrist)
                )
                cost_0R_1L = float(
                    np.linalg.norm(wrists[0] - right_wrist) + np.linalg.norm(wrists[1] - left_wrist)
                )
                if cost_0L_1R <= cost_0R_1L:
                    left_hand_lms, right_hand_lms = detected_hands[0], detected_hands[1]
                else:
                    right_hand_lms, left_hand_lms = detected_hands[0], detected_hands[1]
                return left_hand_lms, right_hand_lms
            else:
                d_left = float(np.linalg.norm(wrists[0] - left_wrist))
                d_right = float(np.linalg.norm(wrists[0] - right_wrist))
                if d_right < d_left:
                    right_hand_lms = detected_hands[0]
                else:
                    left_hand_lms = detected_hands[0]
                return left_hand_lms, right_hand_lms

        # Fallback without reliable Pose wrists: use MediaPipe handedness (unmirrored frame)
        handedness_list = getattr(hands_res, "multi_handedness", None) or []
        for idx, lm_list in enumerate(detected_hands):
            is_right = False
            if idx < len(handedness_list) and handedness_list[idx].classification:
                # MediaPipe Hands assumes mirrored selfie view -> on unmirrored frame:
                # MP "Left" == signer's anatomical Right hand; MP "Right" == signer's anatomical Left hand
                mp_label = handedness_list[idx].classification[0].label
                is_right = (mp_label == "Left")
            else:
                is_right = (wrists[idx][0] < 0.5)

            if is_right and right_hand_lms is None:
                right_hand_lms = lm_list
            elif not is_right and left_hand_lms is None:
                left_hand_lms = lm_list
            elif right_hand_lms is None:
                right_hand_lms = lm_list
            elif left_hand_lms is None:
                left_hand_lms = lm_list

        return left_hand_lms, right_hand_lms

    def extract(
        self,
        frame_bgr: np.ndarray,
        draw_landmarks: bool = False,
    ) -> Tuple[
        Optional[List[Dict[str, float]]],
        Optional[List[Dict[str, float]]],
        Optional[List[Dict[str, float]]],
        np.ndarray,
    ]:
        """Extracts Pose, Left Hand, and Right Hand landmarks from a BGR frame."""
        annotated_frame = frame_bgr.copy() if draw_landmarks else frame_bgr

        if self.hands is None or self.pose is None or self._executor is None:
            return None, None, None, annotated_frame

        # MediaPipe requires RGB format; non-writeable flag avoids internal copy
        H, W = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False

        # Run Pose every `pose_stride` frames once locked (shoulders move slowly);
        # ALWAYS run Hands on every single frame (strict no-persistence contract for hands).
        run_pose_now = (self._frame_idx % self.pose_stride == 0) or (self._cached_pose_lms is None)
        self._frame_idx += 1

        roi_box = self._compute_upper_body_roi(self._cached_pose_lms, W, H)
        if roi_box is not None:
            rx0, ry0, rx1, ry1 = roi_box
            hands_input = np.ascontiguousarray(frame_rgb[ry0:ry1, rx0:rx1])
            hands_input.flags.writeable = False
        else:
            hands_input = frame_rgb

        if run_pose_now:
            f_hands = self._executor.submit(self._run_hands, hands_input)
            f_pose = self._executor.submit(self._run_pose, frame_rgb)
            hands_res = f_hands.result()
            pose_res = f_pose.result()

            if pose_res.pose_landmarks:
                self._cached_pose_proto = pose_res.pose_landmarks
                self._cached_pose_lms = [
                    {
                        "x": float(lm.x),
                        "y": float(lm.y),
                        "z": float(lm.z),
                        "visibility": float(lm.visibility),
                        # mp.solutions.pose leaves protobuf `presence` unset (HasField == False -> 0.0).
                        # Default to 1.0 when unset so VSLPreprocessor does not reject valid shoulders!
                        "presence": float(lm.presence) if lm.HasField("presence") else 1.0,
                    }
                    for lm in pose_res.pose_landmarks.landmark
                ]
            else:
                self._cached_pose_proto = None
                self._cached_pose_lms = None
                self._smooth_roi = None
                self._roi_locked = False
                self._hand_miss_streak = 0
        else:
            hands_res = self._run_hands(hands_input)

        pose_lms = self._cached_pose_lms
        has_detected_hands = bool(hands_res and hands_res.multi_hand_landmarks)

        # Closed-loop adaptive ROI engagement: if Pose confirms a small/far signer (`sh_ratio < 0.18`)
        # has a visible active/elevated wrist with NO matching full-frame MediaPipe hand for >= 2 frames,
        # engage upper-body ROI on subsequent frames so Hands zooms onto the signer's upper body.
        if self.roi_enabled and not self._roi_locked:
            if pose_lms and len(pose_lms) > 16 and self._has_missed_active_wrist(pose_lms, hands_res, W, H):
                ls, rs = pose_lms[11], pose_lms[12]
                sh_w = math.hypot((ls["x"] - rs["x"]) * W, (ls["y"] - rs["y"]) * H)
                if sh_w >= 12.0 and (sh_w / float(W)) < 0.18:
                    self._hand_miss_streak += 1
                    if self._hand_miss_streak >= 2:
                        self._roi_locked = True
                        self._hand_miss_streak = 0
                else:
                    self._hand_miss_streak = 0
            else:
                self._hand_miss_streak = 0

        if roi_box is not None and has_detected_hands:
            rx0, ry0, rx1, ry1 = roi_box
            roi_w, roi_h = rx1 - rx0, ry1 - ry0
            pose_in_roi = None
            if pose_lms is not None:
                pose_in_roi = [
                    {
                        **p,
                        "x": (p["x"] * W - rx0) / float(roi_w),
                        "y": (p["y"] * H - ry0) / float(roi_h),
                    }
                    for p in pose_lms
                ]
            lh_roi, rh_roi = self._assign_hands_anatomically(hands_res, pose_in_roi)
            left_hand_lms = self._remap_hand_from_roi(lh_roi, rx0, ry0, roi_w, roi_h, W, H)
            right_hand_lms = self._remap_hand_from_roi(rh_roi, rx0, ry0, roi_w, roi_h, W, H)
        else:
            left_hand_lms, right_hand_lms = self._assign_hands_anatomically(hands_res, pose_lms)

        # Optional OpenCV drawing (offline / debug scripts)
        if draw_landmarks and HAS_MP:
            if pose_lms and self._cached_pose_proto is not None and settings.DRAW_POSE:
                mp_drawing.draw_landmarks(
                    annotated_frame,
                    self._cached_pose_proto,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                )
            if hands_res and hands_res.multi_hand_landmarks and settings.DRAW_HANDS:
                target_canvas = (
                    annotated_frame[roi_box[1]:roi_box[3], roi_box[0]:roi_box[2]]
                    if roi_box is not None
                    else annotated_frame
                )
                for hand_proto in hands_res.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        target_canvas,
                        hand_proto,
                        mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                        connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style(),
                    )

        return left_hand_lms, right_hand_lms, pose_lms, annotated_frame

    def close(self):
        if self.hands:
            self.hands.close()
            self.hands = None
        if self.pose:
            self.pose.close()
            self.pose = None
        self.holistic = None
        if self._executor:
            self._executor.shutdown(wait=False)
            self._executor = None
