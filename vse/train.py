"""Training entry point.

Usage:
    python -m vse.train --data_dir /path/to/coco2017 [OPTIONS]
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless environments
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from vse.data.vocab import Vocabulary
from vse.data.dataset import CocoDataset, CaptionCollator
from vse.models.combined_model import CombinedModel
from vse.utils.checkpoint import save_checkpoint, load_checkpoint
from vse.utils.config import Config


def build_data_pairs(captions_json_path, train_image_dir):
    """Build a list of (image_path, caption) pairs from the COCO annotations."""
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
    return data_pairs


def train_model(model, dataloader, criterion, optimizer, device, num_epochs=1,
                save_dir=".", start_epoch=0):
    """Run the training loop, saving a checkpoint after every epoch.

    Returns the list of per-epoch average training losses.
    """
    model.to(device)
    train_losses = []

    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        total_loss = 0

        loop = tqdm(dataloader, desc=f"Epoch [{epoch + 1}/{start_epoch + num_epochs}]",
                    leave=False)
        for images, captions in loop:
            images = images.to(device)
            captions = captions.to(device)

            # NOTE: captions are sliced here (captions[:, :-1]) AND again
            # inside DecoderWithAttention.forward, so the input is shortened
            # twice.  This is the behaviour the checkpoint was trained with.
            # See docs/KNOWN_ISSUES.md.
            outputs = model(images, captions[:, :-1])
            targets = captions[:, 1:]

            # Handle possible mismatch
            min_len = min(outputs.size(1), targets.size(1))
            outputs = outputs[:, :min_len, :]
            targets = targets[:, :min_len]

            outputs = outputs.reshape(-1, outputs.shape[2])
            targets = targets.reshape(-1)

            optimizer.zero_grad()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        train_losses.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{start_epoch + num_epochs}] Avg Loss: {avg_loss:.4f}")

        save_checkpoint(model, optimizer, epoch, avg_loss,
                        path=os.path.join(save_dir, "checkpoint_img_caption.pth"))

    # Save loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(start_epoch + 1, start_epoch + num_epochs + 1),
             train_losses, marker="o", color="blue")
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    plt.close()

    return train_losses


def main(argv=None):
    cfg = Config()
    parser = argparse.ArgumentParser(description="Train the image captioning model.")
    parser.add_argument("--data_dir", required=True,
                        help="Root of the COCO 2017 dataset (contains train2017/ and annotations/).")
    parser.add_argument("--save_dir", default=".",
                        help="Directory for checkpoints and loss-curve plot.")
    parser.add_argument("--resume", default=None,
                        help="Path to a checkpoint to resume training from.")
    # NOTE: --num_train_pairs defaults to 60,000, which is the value in the
    # original train.py.  The author's verified figures state 20,000+ images /
    # 90,000+ captions; 60,000 annotation entries correspond to roughly 12,000
    # unique images.  The mismatch is unresolved.
    parser.add_argument("--num_train_pairs", type=int, default=cfg.num_train_pairs,
                        help="Number of (image, caption) pairs to train on.")
    parser.add_argument("--epochs", type=int, default=cfg.num_epochs)
    parser.add_argument("--batch_size", type=int, default=cfg.batch_size)
    parser.add_argument("--lr", type=float, default=cfg.learning_rate)
    parser.add_argument("--embed_size", type=int, default=cfg.embed_size)
    parser.add_argument("--hidden_size", type=int, default=cfg.hidden_size)
    parser.add_argument("--attention_dim", type=int, default=cfg.attention_dim)
    parser.add_argument("--freq_threshold", type=int, default=cfg.freq_threshold)
    parser.add_argument("--device", default=None,
                        help="Device string (default: cuda if available, else cpu).")
    args = parser.parse_args(argv)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # --- Data ---
    train_image_dir = os.path.join(args.data_dir, "train2017")
    captions_json_path = os.path.join(args.data_dir, "annotations", "captions_train2017.json")
    data_pairs = build_data_pairs(captions_json_path, train_image_dir)

    all_captions = [cap for _, cap in data_pairs]
    vocab = Vocabulary(freqthreshold=args.freq_threshold)
    vocab.buildvocabulary(all_captions)

    # Save vocab so inference can load it without the annotations
    os.makedirs(args.save_dir, exist_ok=True)
    vocab.save(os.path.join(args.save_dir, "vocab.json"))

    transform = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(cfg.image_mean), std=list(cfg.image_std)),
    ])

    dataset = CocoDataset(data_pairs, vocab, transform=transform)
    subset_dataset = Subset(dataset, list(range(min(args.num_train_pairs, len(dataset)))))
    collator = CaptionCollator(pad_idx=Vocabulary.PAD_IDX)
    dataloader = DataLoader(subset_dataset, batch_size=args.batch_size, shuffle=True,
                            collate_fn=collator)

    # --- Model ---
    model = CombinedModel(args.embed_size, args.hidden_size, len(vocab),
                          attention_dim=args.attention_dim).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=Vocabulary.PAD_IDX)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 0
    if args.resume:
        start_epoch, _ = load_checkpoint(model, optimizer, args.resume, device=str(device))
        start_epoch += 1  # resume from the next epoch
        print(f"Resumed from epoch {start_epoch}")

    # --- Train ---
    train_model(model, dataloader, criterion, optimizer, device,
                num_epochs=args.epochs, save_dir=args.save_dir, start_epoch=start_epoch)


if __name__ == "__main__":
    main()
