import torch.nn as nn
from torch.optim import AdamW, Muon


def build_optimizers(model: nn.Module, name: str, lr: float, weight_decay: float, adamw_lr: float | None = None) -> list:
    """Pass the uncompiled model so parameter names aren't prefixed with _orig_mod.

    With muon, `lr` drives the Muon matrices and `adamw_lr` (default: `lr`) the embedding and norms.
    """
    if name == "adamw":
        return [AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))]

    # Muon for 2D weights inside the blocks; AdamW for the (tied) embedding and norms.
    muon_params, adamw_params = [], []
    for n, p in model.named_parameters():
        is_hidden_matrix = n.startswith("transformer_blocks.") and p.ndim == 2
        (muon_params if is_hidden_matrix else adamw_params).append(p)

    return [
        # match_rms_adamw scales Muon's update RMS to 0.2 * lr, i.e. AdamW-like.
        Muon(muon_params, lr=lr, weight_decay=weight_decay, adjust_lr_fn="match_rms_adamw"),
        AdamW(adamw_params, lr=adamw_lr or lr, weight_decay=weight_decay, betas=(0.9, 0.95)),
    ]
