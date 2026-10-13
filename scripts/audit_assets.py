import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.config.settings import settings
from backend.app.ml.model_loader import ModelLoader


def audit_assets():
    print("=" * 70)
    print("VSL400 SYSTEM ASSET AUDIT")
    print("=" * 70)

    print(f"Project Root: {settings.PROJECT_ROOT}")
    print(f"Asset Root:   {settings.ASSET_ROOT}")

    # Check files
    files_to_check = [
        ("Checkpoint", settings.CHECKPOINT_PATH),
        ("Label Map", settings.LABEL_MAP_PATH),
        ("Extractor Config", settings.EXTRACTOR_CONFIG_PATH),
    ]

    all_found = True
    for name, path in files_to_check:
        exists = path.exists()
        print(f"[{'FOUND' if exists else 'MISSING'}] {name:16s}: {path}")
        if not exists:
            all_found = False

    if not all_found:
        print("ERROR: One or more critical assets are missing!")
        return False

    # Audit label map
    with open(settings.LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        lmap = json.load(f)
    labels = lmap.get("id_to_label", lmap.get("label_to_id", {}))
    print(f"Label count: {len(labels)} (Expected: 400)")
    assert len(labels) == 400, "Label count must be 400"

    # Audit extractor config
    with open(settings.EXTRACTOR_CONFIG_PATH, "r", encoding="utf-8") as f:
        ext_cfg = json.load(f)
    print(f"Extractor API: {ext_cfg.get('api')}")
    print(f"Pose visibility threshold: {ext_cfg.get('pose_visibility_threshold')}")
    print(f"Clip value: {ext_cfg.get('clip_value')}")

    # Load model and audit state dict
    print("\nLoading checkpoint and validating model...")
    loader = ModelLoader()
    model = loader.load_model(strict=True)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,} (Target: 5,442,961)")
    assert trainable_params == 5442961, "Parameter count mismatch"

    print("\nASSET AUDIT COMPLETED SUCCESSFULLY: 100% PASS")
    print("=" * 70)
    return True


if __name__ == "__main__":
    audit_assets()
