import torch.nn as nn
from torch.optim import AdamW, Muon


def build_optimizers(
    model: nn.Module, name: str, lr: float, weight_decay: float, adamw_lr: float | None = None, lm_head_lr: float | None = None
) -> list:
    """Pass the uncompiled model so parameter names aren't prefixed with _orig_mod.

    With muon, `lr` drives the Muon matrices, `adamw_lr` (default: `lr`) the embedding and norms, and
    `lm_head_lr` (default: `adamw_lr`) the untied lm_head. Weight decay applies to matrices only:
    the embedding's scale is normalised away by the first RMSNorm, and decaying norm gains just shrinks them.
    """
    if name == "adamw":
        return [AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))]

    adamw_lr = adamw_lr or lr
    groups = {"muon": [], "lm_head": [], "other": []}
    for n, p in model.named_parameters():
        if n.startswith("transformer_blocks.") and p.ndim == 2:
            groups["muon"].append(p)
        elif n.startswith("lm_head."):  # absent when tied (v4)
            groups["lm_head"].append(p)
        else:
            groups["other"].append(p)

    adamw_groups = [{"params": groups["other"], "lr": adamw_lr, "weight_decay": 0.0}]
    if groups["lm_head"]:
        adamw_groups.append({"params": groups["lm_head"], "lr": lm_head_lr or adamw_lr, "weight_decay": weight_decay})
    return [
        # match_rms_adamw scales Muon's update RMS to 0.2 * lr, i.e. AdamW-like.
        Muon(groups["muon"], lr=lr, weight_decay=weight_decay, adjust_lr_fn="match_rms_adamw"),
        AdamW(adamw_groups, betas=(0.9, 0.95)),
    ]
