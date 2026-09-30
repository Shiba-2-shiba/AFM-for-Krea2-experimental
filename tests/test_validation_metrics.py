import math
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from afm.metrics import compare_attention_outputs
from scripts import compare_attention_kernels

ROOT = Path(__file__).resolve().parents[1]


def test_numeric_metrics_and_zero_effect():
    reference = torch.tensor([1.0, 3.0])
    candidate = torch.tensor([2.0, 4.0])
    reference_zero = torch.zeros(2)
    candidate_zero = torch.zeros(2)

    result = compare_attention_outputs(candidate, reference, candidate_zero, reference_zero)

    assert result.rel_l2 == pytest.approx(math.sqrt(2.0) / math.sqrt(10.0))
    assert result.scaled_max_abs == pytest.approx(1.0 / 3.0)
    assert result.effect_rel_l2 == pytest.approx(math.sqrt(2.0) / math.sqrt(10.0))
    assert result.effect_resolved is True
    assert result.finite is True


def test_effect_at_three_times_zero_path_error_is_unresolved():
    result = compare_attention_outputs(
        torch.tensor([4.0]),
        torch.tensor([3.0]),
        torch.tensor([1.0]),
        torch.tensor([0.0]),
    )

    assert result.effect_resolved is False


def test_compare_requires_identical_shapes_for_all_four_tensors():
    with pytest.raises(ValueError, match="same shape"):
        compare_attention_outputs(
            torch.zeros(2), torch.zeros(2), torch.zeros(1), torch.zeros(2)
        )


def test_finite_covers_all_four_inputs():
    result = compare_attention_outputs(
        torch.zeros(1), torch.zeros(1), torch.tensor([float("nan")]), torch.zeros(1)
    )
    assert result.finite is False


def test_offline_loader_requests_weights_only(monkeypatch, tmp_path, capsys):
    calls = []

    def fake_load(path, **kwargs):
        calls.append((path, kwargs))
        return torch.ones(1)

    monkeypatch.setattr(compare_attention_kernels.torch, "load", fake_load)
    paths = [tmp_path / name for name in ("candidate.pt", "reference.pt", "cz.pt", "rz.pt")]

    assert compare_attention_kernels.main([str(path) for path in paths]) == 0

    assert len(calls) == 4
    assert all(kwargs == {"map_location": "cpu", "weights_only": True} for _, kwargs in calls)
    assert '"finite": true' in capsys.readouterr().out


def test_compare_script_supports_direct_invocation():
    result = subprocess.run(
        [sys.executable, "scripts/compare_attention_kernels.py", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
