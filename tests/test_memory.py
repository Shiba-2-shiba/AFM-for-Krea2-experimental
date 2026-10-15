from dataclasses import replace

import pytest
import torch

from afm.config import AFMConfig
from afm.memory import ChunkPlan, choose_chunks, estimate_full_joint_logits_mib
from afm.runtime import AFMUnsupportedError, TokenLayout


def _layout(t=5, h=3, w=5):
    return TokenLayout((0, t), (t, t + h * w), (h, w), t + h * w)


def test_full_joint_memory_dimensions():
    assert estimate_full_joint_logits_mib(1, 48, 4096, 512) == pytest.approx(3888.0)


@pytest.mark.parametrize("implementation", ["manual", "augmented"])
def test_choose_chunks_respects_limits_and_full_batch_fixed_allocations(implementation):
    layout = _layout()
    config = AFMConfig(head_chunk_size=4, query_chunk_size=9, max_workspace_mib=1.0)
    plan = choose_chunks((2, 6, 20, 8), (2, 6, 20, 8), layout, config,
                         implementation, torch.float32)
    assert isinstance(plan, ChunkPlan)
    assert 1 <= plan.head_chunk_size <= 4
    assert 1 <= plan.query_chunk_size <= 9
    assert plan.estimated_workspace_mib <= config.max_workspace_mib
    one = choose_chunks((1, 6, 20, 8), (1, 6, 20, 8), layout,
                        replace(config, max_workspace_mib=8), implementation, torch.float32)
    two = choose_chunks((2, 6, 20, 8), (2, 6, 20, 8), layout,
                        replace(config, max_workspace_mib=8), implementation, torch.float32)
    assert two.estimated_workspace_mib > one.estimated_workspace_mib


def test_memory_guard_does_not_discard_fixed_allocations_when_shrinking():
    layout = _layout(t=32, h=8, w=8)
    config = AFMConfig(head_chunk_size=8, query_chunk_size=64, max_workspace_mib=0.01)
    with pytest.raises(AFMUnsupportedError, match="memory_guard"):
        choose_chunks((2, 8, 96, 32), (2, 8, 96, 32), layout,
                      config, "manual", torch.float32)


def test_augmented_memory_includes_math_intermediates_and_retained_tensors():
    layout = TokenLayout((0, 512), (512, 4608), (64, 64), 4608)
    config = AFMConfig(head_chunk_size=2, query_chunk_size=128, max_workspace_mib=1024)
    plan = choose_chunks((1, 48, 4608, 128), (1, 48, 4608, 128), layout,
                         config, "augmented", torch.float16)
    assert plan.head_chunk_size == 2
    assert plan.query_chunk_size == 128
    # Retained QKV + baseline + clone + output alone exceed 300 MiB.
    assert plan.estimated_workspace_mib > 300
    assert 2 * 4096 * 128 * 4 / 2**20 == pytest.approx(4.0)  # FP32 image Q
    assert 2 * 4608 * 256 * 2 / 2**20 == pytest.approx(4.5)  # one augmented K/V


def test_memory_counts_float32_augmented_and_bounded_diagnostic_scratch():
    layout = _layout(t=8, h=4, w=8)
    base = AFMConfig(head_chunk_size=2, query_chunk_size=16, max_workspace_mib=64)
    activation = choose_chunks((2, 4, 40, 8), (2, 4, 40, 8), layout,
                               base, "augmented", torch.float16)
    object.__setattr__(base, "augmented_precision", "float32")
    fp32 = choose_chunks((2, 4, 40, 8), (2, 4, 40, 8), layout,
                         base, "augmented", torch.float16)
    object.__setattr__(base, "spectral_diag", "sampled")
    object.__setattr__(base, "diagnostic_head_limit", 1)
    diagnostic = choose_chunks((2, 4, 40, 8), (2, 4, 40, 8), layout,
                               base, "augmented", torch.float16)
    object.__setattr__(base, "compare_kernels", True)
    compared = choose_chunks((2, 4, 40, 8), (2, 4, 40, 8), layout,
                             base, "augmented", torch.float16)
    assert activation.estimated_workspace_mib < fp32.estimated_workspace_mib
    assert fp32.estimated_workspace_mib < diagnostic.estimated_workspace_mib
    assert diagnostic.estimated_workspace_mib < compared.estimated_workspace_mib


def test_manual_budget_counts_both_fp32_key_and_value_copies():
    layout = _layout()
    config = AFMConfig(head_chunk_size=2, query_chunk_size=7, max_workspace_mib=0.012)
    plan = choose_chunks((1, 2, 20, 8), (1, 2, 20, 8), layout,
                         config, "manual", torch.float16)
    assert plan.head_chunk_size == 1


def test_comparison_budget_counts_float64_metrics_qkv_and_kernel_scratch():
    layout = _layout()
    config = AFMConfig(head_chunk_size=2, query_chunk_size=7, max_workspace_mib=0.02)
    object.__setattr__(config, "compare_kernels", True)
    object.__setattr__(config, "diagnostic_head_limit", 1)
    with pytest.raises(AFMUnsupportedError, match="memory_guard"):
        choose_chunks((1, 2, 20, 8), (1, 2, 20, 8), layout,
                      config, "manual", torch.float16)


def test_augmented_budget_counts_kv_concatenation_temporaries():
    layout = _layout()
    config = AFMConfig(head_chunk_size=1, query_chunk_size=1, max_workspace_mib=0.0102)
    with pytest.raises(AFMUnsupportedError, match="memory_guard"):
        choose_chunks((1, 1, 20, 8), (1, 1, 20, 8), layout,
                      config, "augmented", torch.float16)


def test_augmented_budget_counts_converted_delta_before_query_concat():
    layout = _layout()
    config = AFMConfig(head_chunk_size=1, query_chunk_size=15, max_workspace_mib=0.01505)
    with pytest.raises(AFMUnsupportedError, match="memory_guard"):
        choose_chunks((1, 1, 20, 8), (1, 1, 20, 8), layout,
                      config, "augmented", torch.float16)


@pytest.mark.parametrize("activation_dtype", [torch.float16, torch.bfloat16])
def test_float32_augmented_budget_counts_caller_converted_key_value(activation_dtype):
    layout = _layout()
    config = AFMConfig(
        head_chunk_size=1,
        query_chunk_size=1,
        max_workspace_mib=0.013,
        augmented_precision="float32",
    )
    with pytest.raises(AFMUnsupportedError, match="memory_guard"):
        choose_chunks((1, 1, 20, 8), (1, 1, 20, 8), layout,
                      config, "augmented", activation_dtype)


def test_preserve_text_mass_counts_partition_correction_and_corrected_logits():
    layout = _layout(t=8, h=4, w=8)
    joint = AFMConfig(
        head_chunk_size=2,
        query_chunk_size=16,
        max_workspace_mib=64,
        normalization_mode="joint",
    )
    preserve = replace(joint, normalization_mode="preserve_text_mass")
    joint_plan = choose_chunks(
        (2, 4, 40, 8), (2, 4, 40, 8), layout, joint, "manual", torch.float16
    )
    preserve_plan = choose_chunks(
        (2, 4, 40, 8), (2, 4, 40, 8), layout, preserve, "manual", torch.float16
    )
    assert preserve_plan.estimated_workspace_mib > joint_plan.estimated_workspace_mib


def test_preserve_minimum_workspace_boundary_is_enforced_without_partial_plan():
    layout = _layout(t=2, h=2, w=2)
    config = AFMConfig(
        head_chunk_size=1,
        query_chunk_size=4,
        max_workspace_mib=64,
        normalization_mode="preserve_text_mass",
    )
    minimum = choose_chunks(
        (1, 1, 6, 4), (1, 1, 6, 4), layout, config, "manual", torch.float32
    )
    exact = choose_chunks(
        (1, 1, 6, 4),
        (1, 1, 6, 4),
        layout,
        replace(config, max_workspace_mib=minimum.estimated_workspace_mib),
        "manual",
        torch.float32,
    )
    assert exact == minimum
    with pytest.raises(AFMUnsupportedError, match="memory_guard"):
        choose_chunks(
            (1, 1, 6, 4),
            (1, 1, 6, 4),
            layout,
            replace(config, max_workspace_mib=minimum.estimated_workspace_mib - 1e-9),
            "manual",
            torch.float32,
        )


def test_sequential_diagnostic_heads_share_one_head_peak_workspace():
    layout = _layout(t=8, h=4, w=8)
    one = AFMConfig(
        head_chunk_size=1,
        query_chunk_size=16,
        max_workspace_mib=64,
        spectral_diag="sampled",
        diagnostic_head_limit=1,
    )
    selected = replace(one, diagnostic_head_indices="0,3")
    one_plan = choose_chunks(
        (1, 4, 40, 8), (1, 4, 40, 8), layout, one, "manual", torch.float32
    )
    selected_plan = choose_chunks(
        (1, 4, 40, 8), (1, 4, 40, 8), layout, selected, "manual", torch.float32
    )
    assert selected_plan.head_chunk_size == 1
    # Physical heads are processed sequentially, not retained as a stack.
    assert selected_plan.estimated_workspace_mib == one_plan.estimated_workspace_mib
