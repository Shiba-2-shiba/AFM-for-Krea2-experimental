"""Pure tensor metrics shared by offline and in-server validation."""
from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class ValidationMetrics:
    rel_l2: float
    scaled_max_abs: float
    effect_rel_l2: float
    effect_resolved: bool
    finite: bool


def _norm(tensor: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(tensor.to(dtype=torch.float64)).item())


def compare_attention_outputs(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    candidate_zero: torch.Tensor,
    reference_zero: torch.Tensor,
) -> ValidationMetrics:
    """Compute the acceptance metrics from plan section 6.2.

    All inputs must have exactly the same shape. ``effect_resolved`` is false
    when the reference effect is no larger than the plan's three-times-zero-
    path-error noise floor. Metrics are still returned for diagnostics.
    """
    tensors = (candidate, reference, candidate_zero, reference_zero)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise TypeError("all comparison inputs must be torch.Tensor instances")
    if any(tensor.shape != reference.shape for tensor in tensors):
        raise ValueError("all four comparison tensors must have the same shape")

    finite = all(bool(torch.isfinite(tensor).all().item()) for tensor in tensors)
    candidate64, reference64, candidate_zero64, reference_zero64 = (
        tensor.to(dtype=torch.float64) for tensor in tensors
    )
    difference = candidate64 - reference64
    reference_effect = reference64 - reference_zero64
    candidate_effect = candidate64 - candidate_zero64
    zero_path_error = _norm(candidate_zero64 - reference_zero64)

    rel_l2 = _norm(difference) / max(_norm(reference64), 1e-6)
    if reference64.numel():
        reference_max = float(reference64.abs().max().item())
        max_difference = float(difference.abs().max().item())
    else:
        reference_max = max_difference = 0.0
    scaled_max_abs = max_difference / max(reference_max, 1.0)
    reference_effect_norm = _norm(reference_effect)
    effect_rel_l2 = _norm(candidate_effect - reference_effect) / max(
        reference_effect_norm, 1e-6
    )
    effect_resolved = reference_effect_norm > max(1e-6, 3.0 * zero_path_error)
    finite = finite and all(
        math.isfinite(value) for value in (rel_l2, scaled_max_abs, effect_rel_l2)
    )
    return ValidationMetrics(
        rel_l2=rel_l2,
        scaled_max_abs=scaled_max_abs,
        effect_rel_l2=effect_rel_l2,
        effect_resolved=effect_resolved,
        finite=finite,
    )
