"""Unit tests for CTCDecoder module."""

import pytest
import torch
from backend.app.config.settings import settings
from backend.app.ml.ctc_decoder import CTCDecoder


def test_ctc_decoder_collapse_and_blank_removal():
    decoder = CTCDecoder(label_map_path=settings.LABEL_MAP_PATH, blank_id=400)
    assert len(decoder.id_to_label) == 400

    # Synthetic logits: T=8, C=401
    # We want sequence: Blank, 0, 0, Blank, 1, 1, 1, Blank
    T, C = 8, 401
    logits = torch.zeros(T, C)

    # Class indices to peak at each time step
    target_tokens = [400, 0, 0, 400, 1, 1, 1, 400]
    for t, tok in enumerate(target_tokens):
        logits[t, tok] = 10.0

    glosses, collapsed, raw, blank_ratio = decoder.decode_greedy(logits)

    assert raw == target_tokens
    assert collapsed == [0, 1], f"Expected [0, 1], got {collapsed}"
    assert len(glosses) == 2
    assert glosses[0] == decoder.id_to_label[0]  # "Anh"
    assert glosses[1] == decoder.id_to_label[1]  # "Ba lô"
    assert blank_ratio == 3 / 8  # 3 blank steps out of 8


def test_ctc_decoder_empty_and_all_blank():
    decoder = CTCDecoder(label_map_path=settings.LABEL_MAP_PATH, blank_id=400)

    # All blanks
    logits = torch.zeros(10, 401)
    logits[:, 400] = 5.0

    glosses, collapsed, raw, blank_ratio = decoder.decode_greedy(logits)
    assert glosses == []
    assert collapsed == []
    assert blank_ratio == 1.0


def test_decode_single_gloss_topk():
    decoder = CTCDecoder(label_map_path=settings.LABEL_MAP_PATH, blank_id=400)

    # Synthetic logits where class 42 has highest probability in middle frames
    T, C = 20, 401
    logits = torch.zeros(1, T, C)
    logits[:, :, 400] = 1.0  # background blank
    logits[:, 8:14, 42] = 15.0  # strong peak for class 42

    top5 = decoder.decode_single_gloss_topk(logits, k=5)
    assert len(top5) == 5
    assert top5[0]["label_id"] == 42
    assert top5[0]["rank"] == 1
    assert top5[0]["gloss"] == decoder.id_to_label[42]
    assert top5[0]["diagnostic_confidence"] > top5[1]["diagnostic_confidence"]
    assert top5[0]["diagnostic_confidence"] > 0.9


def test_decode_single_gloss_topk_does_not_inflate_all_blank():
    """All-blank logits must yield near-zero posterior confidence instead of inflated Top-5 softmax."""
    decoder = CTCDecoder(label_map_path=settings.LABEL_MAP_PATH, blank_id=400)

    logits = torch.zeros(1, 20, 401)
    logits[:, :, 400] = 10.0  # overwhelming blank probability
    logits[:, 10, 48] = 3.0   # slight blip for class 48, still overwhelmingly blank

    top5 = decoder.decode_single_gloss_topk(logits, k=5)
    assert len(top5) == 5
    assert top5[0]["label_id"] == 48
    assert top5[0]["diagnostic_confidence"] < 0.05, (
        f"Blank-dominated logits should not report inflated confidence, got {top5[0]['diagnostic_confidence']}"
    )

