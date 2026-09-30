"""Feature detection for the native ComfyUI Krea2 implementation."""
from dataclasses import dataclass
import importlib

import torch
import torch.nn.functional as F

CONFIG_KEY = "krea2_afm_config"
WRAPPER_KEY = "krea2_afm"
SOURCE_COMFY_COMMIT = "9d80841aa1990305cc7a280c5ea505f317efcc2d"


@dataclass(frozen=True)
class Krea2Capabilities:
    patch_size: int
    block_count: int
    query_heads: int
    kv_heads: int
    uses_containers: bool
    model_type: str
    head_dim: int
    sdpa_explicit_scale: bool


def _sdpa_has_explicit_scale() -> bool:
    # The builtin often has no inspectable signature. A tiny CPU probe avoids
    # guessing a PyTorch version and never touches the model or its GPU.
    x = torch.zeros(1, 1, 1, 1)
    try:
        F.scaled_dot_product_attention(x, x, x, scale=1.0)
    except (TypeError, AttributeError, NotImplementedError):
        return False
    return True


def detect_krea2(model_patcher) -> Krea2Capabilities:
    try:
        native = importlib.import_module("comfy.ldm.krea2.model").SingleStreamDiT
        attention = importlib.import_module("comfy.ldm.modules.attention")
        wrappers = importlib.import_module("comfy.patcher_extension").WrappersMP
    except (ImportError, AttributeError) as exc:
        raise ValueError("ComfyUI native Krea2 and wrapper APIs are required") from exc
    for method in ("get_model_object", "clone", "set_model_post_input_patch",
                   "add_wrapper_with_key"):
        if not callable(getattr(model_patcher, method, None)):
            raise ValueError(f"Missing ComfyUI ModelPatcher API: {method}")
    for name in ("OUTER_SAMPLE", "DIFFUSION_MODEL"):
        if not hasattr(wrappers, name):
            raise ValueError(f"Missing ComfyUI wrapper API: {name}")
    if not callable(getattr(attention, "optimized_attention_masked", None)):
        raise ValueError("Missing optimized_attention_masked API")
    model = model_patcher.get_model_object("diffusion_model")
    if not isinstance(model, native):
        raise ValueError("Krea2 AFM requires a native Krea2 SingleStreamDiT model")
    if hasattr(model, "_orig_mod"):
        raise ValueError("torch.compile is not supported by Krea2 AFM diagnostics")
    if (type(getattr(model, "patch", None)) is not int or model.patch <= 0
            or not getattr(model, "blocks", None)):
        raise ValueError("Invalid native Krea2 patch size or blocks")
    for attr in ("txtfusion", "txtmlp", "process_img", "_unpack_context"):
        if not hasattr(model, attr):
            raise ValueError(f"Incomplete native Krea2 structure: {attr}")
    dimensions = []
    for block in model.blocks:
        attn = getattr(block, "attn", None)
        dims = tuple(getattr(attn, name, None) for name in ("heads", "kvheads", "headdim"))
        if any(type(value) is not int or value <= 0 for value in dims):
            raise ValueError("Invalid Krea2 attention head dimensions")
        if dims[0] % dims[1]:
            raise ValueError("Krea2 query heads must be divisible by KV heads")
        dimensions.append(dims)
    if len(set(dimensions)) != 1:
        raise ValueError("Heterogeneous Krea2 attention head dimensions are unsupported")
    heads, kvheads, dim = dimensions[0]
    container = getattr(attention, "AttentionTensorContainer", None)
    uses_containers = container is not None
    if uses_containers and not all(callable(getattr(container, name, None))
                                   for name in ("peek", "take")):
        raise ValueError("Unsupported AttentionTensorContainer API")
    return Krea2Capabilities(model.patch, len(model.blocks), heads, kvheads,
                             uses_containers, type(model).__name__, dim,
                             _sdpa_has_explicit_scale())


def detect_patch_conflicts(model_patcher) -> list[str]:
    opts = model_patcher.model_options.get("transformer_options", {})
    conflicts = []
    if CONFIG_KEY in opts:
        conflicts.append("duplicate Krea2 AFM patch")
    if opts.get("optimized_attention_override") is not None:
        conflicts.append("optimized_attention_override")
    for key in ("post_input", "attn1_patch", "attn1_output_patch", "attn2_patch",
                "attn2_output_patch"):
        if opts.get("patches", {}).get(key):
            conflicts.append(key)
    if opts.get("patches_replace"):
        conflicts.append("patches_replace")
    for wrappers in opts.get("wrappers", {}).values():
        if WRAPPER_KEY in wrappers and CONFIG_KEY not in opts:
            conflicts.append("duplicate Krea2 AFM wrapper")
    if model_patcher.model_options.get("multigpu_thread_pool") is not None:
        conflicts.append("multi-GPU")
    return conflicts
