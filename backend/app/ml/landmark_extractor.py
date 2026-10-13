"""LandmarkExtractor using MediaPipe Holistic and Hands for robust VSL recognition."""

import logging
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np

try:
    import mediapipe as mp
    mp_holistic = mp.solutions.holistic
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    HAS_MP = True
except Exception as e:
    HAS_MP = False
    mp_holistic = None
    mp_hands = None
    mp_pose = None
    mp_drawing = None

from backend.app.config.settings import settings

logger = logging.getLogger("vsl.landmark_extractor")


class LandmarkExtractor:
    """Robust extractor using MediaPipe Holistic as primary with Hands fallback.
    Ensures correct anatomical mapping:
    - left_hand_lms: 21 joints of the signer's anatomical left hand (joints 0..20)
    - right_hand_lms: 21 joints of the signer's anatomical right hand (joints 21..41)
    - pose_lms: 33 joints including shoulders (11, 12) for body-centric normalization
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        if not HAS_MP:
            logger.warning("MediaPipe solutions module not available.")
            self.holistic = None
            self.hands = None
            return

        self.holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            refine_face_landmarks=False,
        )

        # Standalone hands detector as fallback if holistic misses hands
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

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
        """Extracts Pose, Left Hand, and Right Hand landmarks from frame.
        Args:
            frame_bgr: Image as numpy array in BGR format
            draw_landmarks: If True, draws skeleton overlay on frame
        Returns:
            left_hand_lms: 21 landmarks or None
            right_hand_lms: 21 landmarks or None
            pose_lms: 33 landmarks or None
            annotated_frame: Frame with drawn overlays (or original if draw_landmarks=False)
        """
        annotated_frame = frame_bgr.copy() if draw_landmarks else frame_bgr

        if self.holistic is None:
            return None, None, None, annotated_frame

        # MediaPipe requires RGB format
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False

        # 1. Holistic processing (primary)
        results = self.holistic.process(frame_rgb)

        pose_lms: Optional[List[Dict[str, float]]] = None
        left_hand_lms: Optional[List[Dict[str, float]]] = None
        right_hand_lms: Optional[List[Dict[str, float]]] = None

        if results.pose_landmarks:
            pose_lms = [
                {
                    "x": float(lm.x),
                    "y": float(lm.y),
                    "z": float(lm.z),
                    "visibility": float(lm.visibility),
                    "presence": float(getattr(lm, "presence", 1.0)),
                }
                for lm in results.pose_landmarks.landmark
            ]

        if results.left_hand_landmarks:
            left_hand_lms = [
                {"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)}
                for lm in results.left_hand_landmarks.landmark
            ]

        if results.right_hand_landmarks:
            right_hand_lms = [
                {"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)}
                for lm in results.right_hand_landmarks.landmark
            ]

        # 2. Fallback: If Holistic didn't detect one or both hands, query Hands detector
        if (left_hand_lms is None or right_hand_lms is None) and self.hands is not None:
            hands_res = self.hands.process(frame_rgb)
            if hands_res.multi_hand_landmarks:
                for hand_lms in hands_res.multi_hand_landmarks:
                    lm_list = [{"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)} for lm in hand_lms.landmark]
                    wrist_pt = np.array([lm_list[0]["x"], lm_list[0]["y"]], dtype=np.float32)

                    # Geometrically determine if this hand is anatomical Left or Right
                    # In Pose: LM 15 is Left Wrist, LM 16 is Right Wrist
                    is_right = False
                    if pose_lms is not None and len(pose_lms) > 16:
                        left_wrist = np.array([pose_lms[15]["x"], pose_lms[15]["y"]], dtype=np.float32)
                        right_wrist = np.array([pose_lms[16]["x"], pose_lms[16]["y"]], dtype=np.float32)
                        d_left = float(np.linalg.norm(wrist_pt - left_wrist))
                        d_right = float(np.linalg.norm(wrist_pt - right_wrist))
                        is_right = d_right < d_left
                    else:
                        # In unmirrored camera view, user's right hand is on image left (x < 0.5)
                        is_right = (wrist_pt[0] < 0.5)

                    if is_right and right_hand_lms is None:
                        right_hand_lms = lm_list
                    elif not is_right and left_hand_lms is None:
                        left_hand_lms = lm_list

        # 3. Optional OpenCV drawing
        if draw_landmarks and HAS_MP:
            if pose_lms and results.pose_landmarks and settings.DRAW_POSE:
                mp_drawing.draw_landmarks(
                    annotated_frame,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                )
            if left_hand_lms and results.left_hand_landmarks and settings.DRAW_HANDS:
                mp_drawing.draw_landmarks(
                    annotated_frame,
                    results.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                    connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style(),
                )
            if right_hand_lms and results.right_hand_landmarks and settings.DRAW_HANDS:
                mp_drawing.draw_landmarks(
                    annotated_frame,
                    results.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                    connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style(),
                )

        return left_hand_lms, right_hand_lms, pose_lms, annotated_frame

    def close(self):
        if self.holistic:
            self.holistic.close()
        if self.hands:
            self.hands.close()
