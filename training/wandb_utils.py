"""
Checkpoint uploads to W&B that don't stall training.

Files are uploaded in place (policy="immutable", skip_cache=True), so wandb makes no staging or
cache copies of multi-GB checkpoints. Each upload gets its own file under uploads/, deleted once
the upload finishes.
"""
import os
import threading
from pathlib import Path

import wandb


def stage(path: Path, step: int) -> Path:
    """Hardlinks `path` into uploads/ (free: same inode). Safe only because saves replace the
    checkpoint via os.replace, never by writing into the existing file."""
    staged = path.parent / "uploads" / f"{path.stem}-step{step}.pt"
    staged.parent.mkdir(exist_ok=True)
    staged.unlink(missing_ok=True)
    os.link(path, staged)
    return staged


def upload_checkpoint(path: Path, name: str, step: int, keep_history: bool, final: bool = False) -> None:
    """Uploads `path` (a file under uploads/) as artifact `name`, then deletes it locally.

    keep_history=False deletes this run's older versions once the new one is up.
    The final upload blocks so the process doesn't exit mid-upload.
    """
    artifact = wandb.Artifact(name=name, type="model", metadata={"step": step, "run_id": wandb.run.id})
    artifact.add_file(str(path), name=f"{name}.pt", policy="immutable", skip_cache=True)
    aliases = ["latest", f"step-{step}"] + (["final"] if final else [])
    logged = wandb.log_artifact(artifact, aliases=aliases)
    if final:
        _finish(logged, path, keep_history)
    else:
        threading.Thread(target=_finish, args=(logged, path, keep_history), daemon=True).start()


def _finish(logged: wandb.Artifact, path: Path, keep_history: bool) -> None:
    logged.wait()
    path.unlink()
    if keep_history:
        return
    collection = f"{logged.entity}/{logged.project}/{logged.name.split(':')[0]}"
    for old in wandb.Api().artifacts("model", collection):
        if old.metadata.get("run_id") == wandb.run.id and old.version != logged.version:
            old.delete(delete_aliases=True)
