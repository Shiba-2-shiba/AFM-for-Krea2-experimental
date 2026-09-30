"""Compare four offline tensor artifacts without loading executable pickle objects."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from afm.metrics import ValidationMetrics, compare_attention_outputs

__all__ = ["ValidationMetrics", "compare_attention_outputs"]


def _load_tensor(path: Path) -> torch.Tensor:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{path} must contain one tensor")
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate_zero", type=Path)
    parser.add_argument("reference_zero", type=Path)
    args = parser.parse_args(argv)
    metrics = compare_attention_outputs(
        _load_tensor(args.candidate),
        _load_tensor(args.reference),
        _load_tensor(args.candidate_zero),
        _load_tensor(args.reference_zero),
    )
    print(json.dumps(asdict(metrics), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
