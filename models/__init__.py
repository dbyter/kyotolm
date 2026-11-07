from pathlib import Path

import torch

from models import kyotolmv2, kyotolmv4

# Checkpoints record their arch under config["model"]; older ones predate the key and are v2.
MODELS = {"v2": kyotolmv2, "v4": kyotolmv4}


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    """Rebuilds the right architecture from a pretrain or SFT checkpoint, in eval mode on `device`."""
    d = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    tc = d["config"]
    arch = MODELS[tc.get("model", "v2")]
    model = arch.KyotoLM(
        arch.Config(
            vocab_size=d["vocab_size"],
            n_embedding_dim=tc.get("n_embedding_dim", tc.get("embedding_dim", 768)),
            n_head=tc.get("n_heads", 6),
            n_layers=tc.get("n_layers", 12),
        )
    )
    model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in d["model_state_dict"].items()})
    return model.to(device).eval()
