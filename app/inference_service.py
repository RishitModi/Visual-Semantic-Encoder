import argparse
import base64
import io
import math
import re
import threading
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from PIL import Image, UnidentifiedImageError

from utils.checkpoint import load_for_inference
from utils.dataset import build_transform
from utils.visualization import generate_caption_with_attention


class CaptionService:
    """Load an image-captioning model and generate captions with heatmaps."""

    def __init__(self, checkpoint_path: str, vocab_path: str | None = None,
                 device: str = "cpu") -> None:
        self.device = device
        self.model, self.vocab = load_for_inference(checkpoint_path, vocab_path, device)
        self.transform = build_transform()
        self._lock = threading.Lock()

    def predict(self, image_bytes: bytes) -> dict[str, Any]:
        """Return a caption, the display image, and one heatmap per word."""
        with self._lock:
            display_image = self._load_display_image(image_bytes)
            image_tensor = self.transform(display_image)
            words, alphas = generate_caption_with_attention(
                self.model,
                image_tensor,
                self.vocab,
                device=self.device,
            )
            heatmaps = [
                self._create_heatmap(display_image, alpha)
                for alpha in alphas
            ]
            return {
                "caption": " ".join(words),
                "words": words,
                "image": self._png_data_uri(display_image),
                "heatmaps": [self._png_data_uri(image) for image in heatmaps],
            }

    @staticmethod
    def _load_display_image(image_bytes: bytes) -> Image.Image:
        try:
            with Image.open(io.BytesIO(image_bytes)) as source:
                image = source.copy()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise ValueError("Invalid or unsupported image") from exc

        has_transparency = image.mode in {"RGBA", "LA"} or (
            image.mode == "P" and "transparency" in image.info
        )
        if has_transparency:
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, rgba).convert("RGB")
        else:
            image = image.convert("RGB")

        longest_side = max(image.size)
        if longest_side > 640:
            scale = 640 / longest_side
            size = (round(image.width * scale), round(image.height * scale))
            image = image.resize(size, Image.Resampling.LANCZOS)
        return image

    @staticmethod
    def _create_heatmap(display_image: Image.Image, alpha: np.ndarray) -> Image.Image:
        attention = np.asarray(alpha, dtype=np.float32)
        side = int(round(math.sqrt(attention.size)))
        attention = attention.reshape(side, side)
        minimum = float(attention.min())
        maximum = float(attention.max())
        if maximum > minimum:
            attention = (attention - minimum) / (maximum - minimum)
        else:
            attention = np.zeros_like(attention)

        resized = np.asarray(
            Image.fromarray(attention, mode="F").resize(
                display_image.size, Image.Resampling.BILINEAR
            ),
            dtype=np.float32,
        )
        colors = matplotlib.colormaps["jet"](np.clip(resized, 0.0, 1.0))[..., :3]
        base = np.asarray(display_image, dtype=np.float32) / 255.0
        blended = ((base + colors) / 2.0 * 255).astype(np.uint8)
        return Image.fromarray(blended, mode="RGB")

    @staticmethod
    def _png_data_uri(image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"


def _safe_word(word: str) -> str:
    """Make a generated word safe for use in a Windows filename."""
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", word).strip(" .")
    return safe or "word"


def main() -> None:
    """Run caption inference from the command line."""
    parser = argparse.ArgumentParser(description="Generate an image caption and attention heatmaps.")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--vocab", help="Vocabulary path when absent from the checkpoint")
    parser.add_argument("--out", required=True, help="Output folder for heatmaps")
    args = parser.parse_args()

    service = CaptionService(args.checkpoint, args.vocab)
    result = service.predict(Path(args.image).read_bytes())
    print(result["caption"])

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, (word, data_uri) in enumerate(zip(result["words"], result["heatmaps"])):
        encoded = data_uri.split(",", 1)[1]
        output_path = output_dir / f"heatmap_{index}_{_safe_word(word)}.png"
        output_path.write_bytes(base64.b64decode(encoded))


if __name__ == "__main__":
    main()
