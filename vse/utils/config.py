"""Default configuration dataclass.

All defaults match the hyperparameters used in the original training run.
"""

from dataclasses import dataclass


@dataclass
class Config:
    """Training and model configuration with defaults from the original run."""

    # Model
    embed_size: int = 256
    hidden_size: int = 512
    attention_dim: int = 256

    # Data
    image_size: int = 224
    freq_threshold: int = 5
    max_len: int = 20
    # NOTE: The default of 60000 (image, caption) pairs is the value hardcoded
    # in the original train.py.  The author's verified figures state 20,000+
    # images and 90,000+ captions.  Because each COCO image has ~5 captions,
    # 60,000 annotation-list entries correspond to roughly 12,000 unique images,
    # not 20,000+.  The mismatch is unresolved; see docs/KNOWN_ISSUES.md.
    num_train_pairs: int = 60000

    # Training
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_epochs: int = 60

    # ImageNet normalisation (used in the original run)
    image_mean: tuple = (0.485, 0.456, 0.406)
    image_std: tuple = (0.229, 0.224, 0.225)
