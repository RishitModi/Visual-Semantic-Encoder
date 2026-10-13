"""Checkpoint save/load utilities.

Checkpoint format (matches the original training run):
    {
        "epoch": int,
        "model_state_dict": OrderedDict,
        "optimizer_state_dict": OrderedDict,
        "loss": float,
    }
"""

import torch


def save_checkpoint(model, optimizer, epoch, loss, path="checkpoint_img_caption.pth"):
    """Save model and optimizer state to *path*."""
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }, path)


def load_checkpoint(model, optimizer, path="checkpoint_img_caption.pth", device="cpu"):
    """Load a checkpoint saved by :func:`save_checkpoint`.

    Returns ``(epoch, loss)`` from the stored checkpoint.
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"], checkpoint["loss"]
