import argparse
from pathlib import Path

import torch
from PIL import Image

from utils.checkpoint import load_for_inference
from utils.dataset import build_transform
from utils.visualization import generate_caption_with_attention, visualize_attention


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a caption (and attention maps) for a single image.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Trained checkpoint path.")
    parser.add_argument("--vocab", type=str, default=None,
                        help="vocab.txt written during training. Only needed if the checkpoint has no vocabulary inside it.")
    parser.add_argument("--no-plot", action="store_true", help="Only print the caption, do not open the attention-map window.")
    parser.add_argument("--save", type=str, default=None, help="Also save the attention-map figure to this image file.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = Path(args.image)
    checkpoint_path = Path(args.checkpoint)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model, vocab = load_for_inference(checkpoint_path, args.vocab, device=args.device)

    image_tensor = build_transform()(Image.open(image_path).convert("RGB"))
    caption, alphas = generate_caption_with_attention(model, image_tensor, vocab, device=args.device)
    print("Predicted:", " ".join(caption))

    if not args.no_plot:
        visualize_attention(image_tensor, caption, alphas, save_path=args.save)


if __name__ == "__main__":
    main()
