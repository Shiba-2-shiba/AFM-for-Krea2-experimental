from copy import deepcopy

import pytest

from afm.config import AFMConfig
from afm.runtime import AFMUnsupportedError
from afm.selection import select_implementation, load_qualified_profiles


def signature():
    return {"python": "3.10.11", "pytorch": "2.10.0", "cuda": "12.8",
            "gpu": "example", "compute_capability": [8, 6], "device_type": "cuda",
            "comfyui_commit": "fixture", "baseline_backend": "torch",
            "sdpa_settings": {"math": True, "flash": True}, "activation_dtype": "torch.float32",
            "original_head_dim": 8, "sdpa_explicit_scale": True,
            "filter_signature": {"version": "radial_v1", "cutoff": .25, "soft_mask": True,
                                 "mask_width": .05, "preserve_dc": True, "schedule": "curve"},
            "text_tokens": 5, "image_tokens": 15, "batch": 2, "heads": 4,
            "grid_h": 3, "grid_w": 5, "head_chunk_size": 2, "query_chunk_size": 128,
            "strength": .05}


def profile():
    sig = signature()
    ranges = {}
    for key in ("text_tokens", "image_tokens", "batch", "heads", "grid_h", "grid_w",
                "head_chunk_size", "query_chunk_size", "strength"):
        value = sig.pop(key)
        ranges[key] = [value, value]
    return {"id": "fixture-only", "qualified": True, "signature": sig, "ranges": ranges,
            "evidence": {"numerical_report": "fixture-numerics.json", "benchmark_manifest": "fixture-bench.json"}}


def test_auto_unqualified_uses_manual():
    assert load_qualified_profiles() == []
    choice = select_implementation(AFMConfig(mode="edit"), signature(), [])
    assert choice.implementation == "manual"
    assert choice.reason == "profile_not_qualified"


def test_auto_qualified_profile():
    choice = select_implementation(AFMConfig(mode="edit"), signature(), [profile()])
    assert choice.implementation == "augmented"
    assert choice.profile_id == "fixture-only"


@pytest.mark.parametrize("key,value", [("gpu", "other"), ("activation_dtype", "torch.bfloat16"),
                                       ("comfyui_commit", "changed"), ("text_tokens", 6),
                                       ("image_tokens", 16), ("batch", 1), ("grid_h", 5),
                                       ("head_chunk_size", 1), ("strength", .2), ("cuda", None)])
def test_auto_profile_scope_exact(key, value):
    sig = signature()
    sig[key] = value
    assert select_implementation(AFMConfig(mode="edit"), sig, [profile()]).implementation == "manual"


@pytest.mark.parametrize("change", ["evidence", "qualified", "missing_range", "missing_signature"])
def test_incomplete_profiles_fail_closed(change):
    item = deepcopy(profile())
    if change == "evidence":
        item["evidence"] = {}
    elif change == "qualified":
        item["qualified"] = False
    elif change == "missing_range":
        item["ranges"].pop("batch")
    else:
        item["signature"].pop("pytorch")
    assert select_implementation(AFMConfig(mode="edit"), signature(), [item]).implementation == "manual"


@pytest.mark.parametrize("zero", ["original", "manual", "augmented"])
def test_zero_control_precedence(zero):
    cfg = AFMConfig(mode="edit", strength=0, implementation="augmented", zero_strength_mode=zero)
    assert select_implementation(cfg, signature(), []).implementation == zero


def test_forced_augmented_requires_scale_support():
    sig = signature()
    sig["sdpa_explicit_scale"] = False
    with pytest.raises(AFMUnsupportedError, match="unsupported_augmented_backend"):
        select_implementation(AFMConfig(mode="edit", implementation="augmented"), sig, [])
    assert select_implementation(AFMConfig(mode="edit"), sig, []).implementation == "manual"


def test_auto_matches_executed_chunks_after_memory_reduction():
    from dataclasses import replace
    import torch
    from afm.backend import _select_edit_plan
    from afm.memory import choose_chunks
    from afm.runtime import TokenLayout
    layout = TokenLayout((0, 32), (32, 96), (8, 8), 96)
    shape = (2, 8, 96, 32)
    smallest = choose_chunks(shape, shape, layout,
                             AFMConfig(mode="edit", head_chunk_size=1, query_chunk_size=32),
                             "augmented", torch.float32)
    config = AFMConfig(mode="edit", head_chunk_size=8, query_chunk_size=64,
                       max_workspace_mib=smallest.estimated_workspace_mib + 1e-5)
    actual = choose_chunks(shape, shape, layout, config, "augmented", torch.float32)
    assert (actual.head_chunk_size, actual.query_chunk_size) != (8, 64)
    sig = signature()
    sig.update(text_tokens=32, image_tokens=64, grid_h=8, grid_w=8, batch=2,
               heads=8, original_head_dim=32, head_chunk_size=8, query_chunk_size=64)
    item = profile()
    item["signature"] = {key: value for key, value in sig.items() if key not in item["ranges"]}
    item["ranges"] = {key: [sig[key], sig[key]] for key in item["ranges"]}
    item["ranges"]["head_chunk_size"] = [actual.head_chunk_size, actual.head_chunk_size]
    item["ranges"]["query_chunk_size"] = [actual.query_chunk_size, actual.query_chunk_size]
    choice, chunks = _select_edit_plan(config, sig, [item], shape, shape, layout, torch.float32)
    assert choice.implementation == "augmented"
    assert chunks == actual
    # A profile measured only at the requested (larger) chunks must NOT authorize
    # execution with the shrunken plan.
    item["ranges"]["head_chunk_size"] = [8, 8]
    item["ranges"]["query_chunk_size"] = [64, 64]
    choice, _ = _select_edit_plan(config, sig, [item], shape, shape, layout, torch.float32)
    assert choice.implementation == "manual"
