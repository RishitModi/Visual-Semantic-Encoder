"""Generate attention heatmaps for a single image.

Usage:
    python -m vse.attention --image path/to/image.jpg \
                            --checkpoint path/to/checkpoint.pth \
                            --vocab path/to/vocab.json \
                            [--save path/to/output.png]
"""

import argparse

import torch
import torchvision.transforms as transforms
from PIL import Image

from vse.data.vocab import Vocabulary
from vse.models.combined_model import CombinedModel
from vse.utils.checkpoint import load_checkpoint
from vse.utils.visualization import generate_caption_with_attention, visualize_attention
from vse.utils.config import Config


def main(argv=None):
    cfg = Config()
    parser = argparse.ArgumentParser(description="Visualise attention heatmaps for an image.")
    parser.add_argument("--image", required=True, help="Path to an input image.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
    parser.add_argument("--vocab", required=True, help="Path to vocab.json.")
    parser.add_argument("--save", default=None,
                        help="If provided, save the figure to this path instead of displaying it.")
    parser.add_argument("--max_len", type=int, default=cfg.max_len)
    parser.add_argument("--embed_size", type=int, default=cfg.embed_size)
    parser.add_argument("--hidden_size", type=int, default=cfg.hidden_size)
    parser.add_argument("--attention_dim", type=int, default=cfg.attention_dim)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    vocab = Vocabulary.load(args.vocab)

    model = CombinedModel(args.embed_size, args.hidden_size, len(vocab),
                          attention_dim=args.attention_dim).to(device)
    load_checkpoint(model, optimizer=None, path=args.checkpoint, device=str(device))

    transform = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(cfg.image_mean), std=list(cfg.image_std)),
    ])

    img = Image.open(args.image).convert("RGB")
    img_tensor = transform(img)

    caption, alphas = generate_caption_with_attention(
        model, img_tensor, vocab, max_len=args.max_len, device=str(device)
    )
    print("Caption:", " ".join(caption))
    visualize_attention(img_tensor, caption, alphas, save_path=args.save)


if __name__ == "__main__":
    main()
