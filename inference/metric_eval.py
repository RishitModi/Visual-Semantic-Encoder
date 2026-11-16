import argparse
from pathlib import Path

import torch

from utils.checkpoint import load_for_inference
from utils.dataset import build_transform, load_coco_pairs
from utils.evaluation import evaluate_bleu


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a captioning model with BLEU scores on COCO val2017.")
    parser.add_argument("--dataset-dir", type=str, required=True, help="COCO root containing val2017/ and annotations/captions_val2017.json")
    parser.add_argument("--checkpoint", type=str, required=True, help="Trained checkpoint path.")
    parser.add_argument("--vocab", type=str, default=None, help="vocab.txt, only needed if the checkpoint has no vocabulary inside it.")
    parser.add_argument("--num-images", type=int, default=500, help="Number of validation images to score.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    model, vocab = load_for_inference(args.checkpoint, args.vocab, device=args.device)
    val_pairs = load_coco_pairs(args.dataset_dir, "val2017")
    bleu1, bleu4 = evaluate_bleu(model, vocab, val_pairs, build_transform(), args.device, args.num_images)
    print(f"BLEU-1: {bleu1:.4f}, BLEU-4: {bleu4:.4f}")


if __name__ == "__main__":
    main()
