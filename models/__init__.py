from pathlib import Path

import torch

from models import kyotolmv2, kyotolmv4, kyotolmv5

# Checkpoints record their arch under config["model"]; older ones predate the key and are v2.
MODELS = {"v2": kyotolmv2, "v4": kyotolmv4, "v5": kyotolmv5}


def resolve_checkpoint(ref: str | Path) -> Path:
    """A local path, or a W&B model artifact such as 'lm:step-20000' or 'sft:latest' (project
    kyotolm unless the ref names one). Inside a wandb run the download is recorded as an input."""
    if Path(ref).expanduser().is_file():
        return Path(ref).expanduser()
    import wandb

    ref = str(ref)
    if wandb.run:
        artifact = wandb.run.use_artifact(ref, type="model")  # resolved against the run's project
    else:
        artifact = wandb.Api().artifact(ref if "/" in ref else f"kyotolm/{ref}", type="model")
    return next(Path(artifact.download()).glob("*.pt"))


def load_model(checkpoint: str | Path, device: torch.device) -> torch.nn.Module:
    """Rebuilds the right architecture from a pretrain or SFT checkpoint (path or W&B artifact),
    in eval mode on `device`."""
    d = torch.load(resolve_checkpoint(checkpoint), map_location="cpu", weights_only=False)
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
