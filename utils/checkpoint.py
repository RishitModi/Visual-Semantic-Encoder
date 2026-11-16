import os
import warnings

import torch

from models.combined_model import CombinedModel
from .vocab import Vocabulary


def save_checkpoint(model, optimizer, epoch, loss, path, vocab=None):
    """Save a training checkpoint. If a vocabulary is given it is stored inside the file too."""
    path = str(path)
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    if vocab is not None:
        payload["vocab_itos"] = dict(vocab.itos)
    tmp_path = path + ".tmp"  # write to a temp file first so a crash cannot corrupt the real checkpoint
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(model, optimizer=None, path="checkpoint.pth"):
    """Load weights (and, if possible, the optimizer state) to resume training."""
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except ValueError:
            warnings.warn("Optimizer state could not be restored (parameter groups differ); "
                          "continuing with a fresh optimizer.")
    return checkpoint["epoch"], checkpoint["loss"]


def load_for_inference(checkpoint_path, vocab_path=None, device="cpu"):
    """Rebuild the model and vocabulary from a checkpoint.

    Model sizes are read from the saved weights. The vocabulary comes from the checkpoint if it
    contains one, otherwise from vocab_path (the vocab.txt written during training).
    """
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    state = checkpoint["model_state_dict"]
    vocab_size = state["decoder.fc.weight"].shape[0]
    embed_size = state["decoder.embedding.weight"].shape[1]
    hidden_size = state["decoder.lstm.weight_hh"].shape[1]
    attention_dim = state["decoder.attention.encoder_att.weight"].shape[0]

    if "vocab_itos" in checkpoint:
        vocab = Vocabulary.from_itos(checkpoint["vocab_itos"])
    elif vocab_path is not None:
        vocab = Vocabulary.load(vocab_path)
    else:
        raise ValueError("This checkpoint has no vocabulary inside it. Pass --vocab path/to/vocab.txt")
    if len(vocab) != vocab_size:
        raise ValueError(f"The vocabulary has {len(vocab)} words but the model expects {vocab_size}. "
                         "Use the vocab.txt that was created together with this checkpoint.")

    model = CombinedModel(embed_size, hidden_size, vocab_size, attention_dim).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, vocab
