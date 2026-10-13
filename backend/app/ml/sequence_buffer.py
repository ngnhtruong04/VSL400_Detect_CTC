"""Rolling temporal sequence buffer for realtime sign language inference."""

from collections import deque
from typing import Optional
import numpy as np
import torch


class RealtimeSequenceBuffer:
    """Maintains a rolling window of recent frame feature vectors."""

    def __init__(
        self,
        max_frames: int = 90,
        min_frames: int = 15,
        inference_stride: int = 2,
        feature_dim: int = 294,
    ):
        self.max_frames = max_frames
        self.min_frames = min_frames
        self.inference_stride = inference_stride
        self.feature_dim = feature_dim

        self.buffer = deque(maxlen=max_frames)
        self.frame_counter: int = 0

    def append(self, frame_features: np.ndarray):
        """Appends a 294-dim frame feature vector to the rolling buffer."""
        if frame_features.shape != (self.feature_dim,):
            raise ValueError(
                f"Expected frame feature shape ({self.feature_dim},), got {frame_features.shape}"
            )
        self.buffer.append(frame_features)
        self.frame_counter += 1

    def should_infer(self) -> bool:
        """Determines whether inference should run on the current frame."""
        if len(self.buffer) < self.min_frames:
            return False
        return (self.frame_counter % self.inference_stride) == 0

    def get_sequence_tensor(self, device: torch.device) -> torch.Tensor:
        """Returns the current buffer sequence as a PyTorch tensor with shape (1, T, 294)."""
        seq_np = np.stack(list(self.buffer), axis=0).astype(np.float32)  # (T, 294)
        tensor = torch.from_numpy(seq_np).unsqueeze(0).to(device)       # (1, T, 294)
        return tensor

    def trim_to_tail(self, n_tail: int = 5):
        """Keeps only the most recent n_tail frames, discarding stale idle frames."""
        while len(self.buffer) > n_tail:
            self.buffer.popleft()

    def clear(self):
        """Clears all frames from buffer."""
        self.buffer.clear()
        self.frame_counter = 0

    def __len__(self) -> int:
        return len(self.buffer)

