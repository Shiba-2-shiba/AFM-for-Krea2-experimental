import torch

from afm.spectral import edit_image_text_logits, schedule_alphas


def dense_joint_image_rows(q, k, v, layout, config, progress, scale):
    """Small CPU oracle; production kernels must not call this."""
    logits = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    image = slice(*layout.image_slice)
    text = slice(*layout.text_slice)
    alpha_lf, alpha_hf = schedule_alphas(config, progress.progress)
    logits[..., image, text] = edit_image_text_logits(
        logits[..., image, text], layout.grid, alpha_lf, alpha_hf, config
    )
    probabilities = torch.softmax(logits[..., image, :], dim=-1)
    return torch.matmul(probabilities, v.float()).to(v.dtype)
