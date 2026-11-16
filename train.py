import argparse
import os
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save the loss plot to a file; a window would block the script
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.optim as optim  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402
from tqdm import tqdm  # noqa: E402

from models.combined_model import CombinedModel  # noqa: E402
from utils.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from utils.dataset import CocoDataset, build_transform, collate_fn, load_coco_pairs  # noqa: E402
from utils.vocab import Vocabulary  # noqa: E402

EMBED_SIZE, HIDDEN_SIZE, ATTENTION_DIM = 256, 512, 256


def train_model(model, dataloader, criterion, optimizer, vocab, device, num_epochs,
                start_epoch=0, checkpoint_path=None, plot_path=None):
    model.to(device)
    train_losses = []

    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0.0

        loop = tqdm(dataloader, desc=f"Epoch [{epoch + 1}/{num_epochs}]", leave=False)
        for images, captions in loop:
            images = images.to(device)
            captions = captions.to(device)

            # The decoder shifts the captions itself (its inputs are captions[:, :-1]),
            # so pass the full captions here and compare against captions[:, 1:].
            outputs = model(images, captions)
            targets = captions[:, 1:]
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / max(len(dataloader), 1)
        train_losses.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}] Avg Loss: {avg_loss:.4f}")

        if checkpoint_path is not None:
            save_checkpoint(model, optimizer, epoch, avg_loss, checkpoint_path, vocab=vocab)

    if train_losses and plot_path is not None:
        plt.figure(figsize=(8, 5))
        plt.plot(range(start_epoch + 1, start_epoch + len(train_losses) + 1), train_losses, marker="o", color="blue")
        plt.title("Training Loss over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()
        print(f"Loss curve saved to {plot_path}")

    return train_losses


def build_vocab_and_dataloader(dataset_dir, batch_size=64, subset_size=100000, freq_threshold=5, num_workers=0):
    data_pairs = load_coco_pairs(dataset_dir, "train2017")

    vocab = Vocabulary(freqthreshold=freq_threshold)
    vocab.buildvocabulary([caption for _, caption in data_pairs])

    dataset = CocoDataset(data_pairs, vocab, transform=build_transform())
    use = min(subset_size, len(dataset)) if subset_size and subset_size > 0 else len(dataset)

    dataloader = DataLoader(
        Subset(dataset, list(range(use))),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn, vocab=vocab),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    print(f"Training on {use} captions ({len(dataloader)} batches per epoch), vocabulary size {len(vocab)}")
    return vocab, dataloader


def main():
    parser = argparse.ArgumentParser(description="Train an image captioning model.")
    parser.add_argument("--dataset-dir", type=str, default="data/coco2017", help="COCO root containing train2017/ and annotations/")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size.")
    parser.add_argument("--subset-size", type=int, default=100000, help="Number of training captions to use. 0 = use all.")
    parser.add_argument("--lr", type=float, default=4e-4, help="Learning rate.")
    parser.add_argument("--freq-threshold", type=int, default=5, help="Minimum word frequency for the vocabulary.")
    parser.add_argument("--num-workers", type=int, default=0 if os.name == "nt" else 2, help="DataLoader worker processes.")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_custom", help="Where checkpoint_img_caption.pth and vocab.txt are written.")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to continue training from.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "checkpoint_img_caption.pth"
    if checkpoint_path.exists() and not args.resume:
        raise FileExistsError(
            f"{checkpoint_path} already exists; pass --resume to continue training instead of overwriting it."
        )

    vocab, dataloader = build_vocab_and_dataloader(
        dataset_dir=args.dataset_dir,
        batch_size=args.batch_size,
        subset_size=args.subset_size,
        freq_threshold=args.freq_threshold,
        num_workers=args.num_workers,
    )
    vocab_path = checkpoint_dir / "vocab.txt"
    vocab.save(vocab_path)
    print(f"Vocabulary saved to {vocab_path}")

    model = CombinedModel(EMBED_SIZE, HIDDEN_SIZE, len(vocab), ATTENTION_DIM).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"])
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    start_epoch = 0
    if args.resume:
        last_epoch, last_loss = load_checkpoint(model, optimizer, args.resume)
        start_epoch = last_epoch + 1
        print(f"Resuming after epoch {last_epoch + 1} (loss {last_loss:.4f})")

    train_model(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        optimizer=optimizer,
        vocab=vocab,
        device=device,
        num_epochs=args.epochs,
        start_epoch=start_epoch,
        checkpoint_path=str(checkpoint_path),
        plot_path=str(checkpoint_dir / "loss_curve.png"),
    )
    print(f"Done. Model: {checkpoint_path}   Vocabulary: {vocab_path}")


if __name__ == "__main__":
    main()
