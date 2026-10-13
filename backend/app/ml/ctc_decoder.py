"""CTC Greedy Decoder for Vietnamese Sign Language (VSL400).

Decodes model logits into Vietnamese Gloss sequences:
1. Argmax over vocabulary dimension
2. Collapse consecutive repeating tokens
3. Remove CTC Blank tokens (ID 400)
4. Map token IDs to Vietnamese Gloss labels
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Union, Any
import torch


class CTCDecoder:
    def __init__(
        self,
        label_map_path: Union[str, Path],
        blank_id: int = 400,
    ):
        self.blank_id = blank_id
        self.label_map_path = Path(label_map_path)
        self.label_to_id: Dict[str, int] = {}
        self.id_to_label: Dict[int, str] = {}
        self._load_label_map()

    def _load_label_map(self):
        if not self.label_map_path.exists():
            raise FileNotFoundError(f"Label map not found at {self.label_map_path}")

        with open(self.label_map_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle various formats: {"label_to_id": ..., "id_to_label": ...} or direct dict
        if "label_to_id" in data:
            self.label_to_id = data["label_to_id"]
        if "id_to_label" in data:
            self.id_to_label = {int(k): v for k, v in data["id_to_label"].items()}
        else:
            self.id_to_label = {v: k for k, v in self.label_to_id.items()}

        if len(self.id_to_label) != 400:
            raise ValueError(f"Expected exactly 400 labels, found {len(self.id_to_label)}")

    def decode_greedy(
        self,
        logits: torch.Tensor,
    ) -> Tuple[List[str], List[int], List[int], float]:
        """Performs greedy CTC decoding on logits.
        Args:
            logits: Tensor of shape (T, C) or (1, T, C)
        Returns:
            gloss_sequence: List of predicted Vietnamese Gloss strings
            collapsed_tokens: List of token IDs after collapsing and removing blank
            raw_tokens: Raw argmax token IDs for all time steps
            blank_ratio: Ratio of time steps where the blank token was predicted
        """
        if logits.dim() == 3:
            logits = logits.squeeze(0)  # (T, C)

        if logits.size(0) == 0:
            return [], [], [], 1.0

        # Step 1: Argmax along class dimension
        raw_tokens = torch.argmax(logits, dim=-1).tolist()

        # Calculate blank ratio
        blank_count = sum(1 for tok in raw_tokens if tok == self.blank_id)
        blank_ratio = blank_count / len(raw_tokens) if len(raw_tokens) > 0 else 1.0

        # Step 2: Collapse consecutive repeats
        collapsed_raw: List[int] = []
        prev_tok = None
        for tok in raw_tokens:
            if tok != prev_tok:
                collapsed_raw.append(tok)
                prev_tok = tok

        # Step 3: Remove CTC Blank token
        collapsed_tokens = [tok for tok in collapsed_raw if tok != self.blank_id]

        # Step 4: Map token IDs to Vietnamese Gloss labels
        gloss_sequence = [
            self.id_to_label[tok]
            for tok in collapsed_tokens
            if tok in self.id_to_label
        ]

        return gloss_sequence, collapsed_tokens, raw_tokens, blank_ratio

    def decode_single_gloss_topk(
        self,
        logits: torch.Tensor,
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Computes single-token CTC diagnostic scores for all 400 hypotheses.
        Returns Top-K candidate glosses with rank, label_id, score, and relative diagnostic confidence.
        Note: Confidence is diagnostic/relative conditional confidence, not calibrated absolute probability.
        """
        if logits.dim() == 2:
            logits = logits.unsqueeze(0)  # (1, T, C)

        B, T, C = logits.shape
        if T == 0:
            return []

        num_classes = min(C - 1, 400)
        # Log softmax: (T, 1, C) for PyTorch CTCLoss
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1).transpose(0, 1)

        # Batch replicate across all candidate classes
        log_probs_exp = log_probs.repeat(1, num_classes, 1)
        targets = torch.arange(num_classes, dtype=torch.long, device=logits.device)
        input_lengths = torch.full((num_classes,), T, dtype=torch.long, device=logits.device)
        target_lengths = torch.ones(num_classes, dtype=torch.long, device=logits.device)

        # Negative CTC loss gives log P(class | sequence)
        ctc_loss = torch.nn.functional.ctc_loss(
            log_probs_exp,
            targets,
            input_lengths,
            target_lengths,
            blank=self.blank_id,
            reduction="none",
            zero_infinity=True,
        )
        log_p = -ctc_loss

        # Extract Top-K
        actual_k = min(k, num_classes)
        topk_scores, topk_indices = torch.topk(log_p, actual_k)

        # Softmax over Top-K for relative diagnostic confidence
        relative_probs = torch.nn.functional.softmax(topk_scores, dim=-1).tolist()
        scores_list = topk_scores.tolist()
        indices_list = topk_indices.tolist()

        candidates = []
        for rank, (idx, score, prob) in enumerate(zip(indices_list, scores_list, relative_probs), start=1):
            candidates.append({
                "rank": rank,
                "label_id": int(idx),
                "gloss": self.id_to_label.get(int(idx), f"GLOSS_{idx}"),
                "log_score": round(float(score), 4),
                "diagnostic_confidence": round(float(prob), 4),
            })
        return candidates

