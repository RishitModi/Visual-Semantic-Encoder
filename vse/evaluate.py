"""Evaluate BLEU scores on a subset of the training data.

Usage:
    python -m vse.evaluate --checkpoint path/to/checkpoint.pth \
                           --vocab path/to/vocab.json \
                           --data_dir /path/to/coco2017 [OPTIONS]

NOTE: This script evaluates on a subset of the *training* data (not a
held-out split), with a single reference caption per image and greedy
decoding.  See docs/KNOWN_ISSUES.md for caveats.
"""

import argparse
import json
import os

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from vse.data.vocab import Vocabulary
from vse.data.dataset import CocoDataset, CaptionCollator
from vse.models.combined_model import CombinedModel
from vse.utils.checkpoint import load_checkpoint
from vse.utils.evaluation import evaluate_bleu
from vse.utils.config import Config


def main(argv=None):
    cfg = Config()
    parser = argparse.ArgumentParser(description="Evaluate BLEU scores.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
    parser.add_argument("--vocab", required=True, help="Path to vocab.json.")
    parser.add_argument("--data_dir", required=True,
                        help="Root of the COCO 2017 dataset.")
    parser.add_argument("--num_eval", type=int, default=1000,
                        help="Number of items to evaluate on (from the start of the dataset).")
    parser.add_argument("--batch_size", type=int, default=cfg.batch_size)
    parser.add_argument("--embed_size", type=int, default=cfg.embed_size)
    parser.add_argument("--hidden_size", type=int, default=cfg.hidden_size)
    parser.add_argument("--attention_dim", type=int, default=cfg.attention_dim)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    vocab = Vocabulary.load(args.vocab)

    # --- Data ---
    train_image_dir = os.path.join(args.data_dir, "train2017")
    captions_json_path = os.path.join(args.data_dir, "annotations", "captions_train2017.json")
    with open(captions_json_path, "r", encoding="utf-8") as f:
        captions_json = json.load(f)
    id_to_filename = {img["id"]: img["file_name"] for img in captions_json["images"]}
    data_pairs = []
    for ann in captions_json["annotations"]:
        img_id = ann["image_id"]
        if img_id in id_to_filename:
            data_pairs.append(
                (os.path.join(train_image_dir, id_to_filename[img_id]), ann["caption"])
            )

    transform = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(cfg.image_mean), std=list(cfg.image_std)),
    ])

    dataset = CocoDataset(data_pairs, vocab, transform=transform)
    subset = Subset(dataset, list(range(min(args.num_eval, len(dataset)))))
    collator = CaptionCollator(pad_idx=Vocabulary.PAD_IDX)
    dataloader = DataLoader(subset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collator)

    # --- Model ---
    model = CombinedModel(args.embed_size, args.hidden_size, len(vocab),
                          attention_dim=args.attention_dim).to(device)
    load_checkpoint(model, optimizer=None, path=args.checkpoint, device=str(device))

    # --- Evaluate ---
    bleu1, bleu4 = evaluate_bleu(model, dataloader, vocab, device=device)
    print(f"BLEU-1: {bleu1:.4f}, BLEU-4: {bleu4:.4f}")


if __name__ == "__main__":
    main()
