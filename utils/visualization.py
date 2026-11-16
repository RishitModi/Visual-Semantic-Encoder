import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def _default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def generate_caption_with_attention(model, image, vocab, max_len=20, device=None):
    """Greedy decoding. Returns the caption (list of words) and one attention map per word."""
    device = device or _default_device()
    model.eval()
    encoder_out = model.encoder(image.unsqueeze(0).to(device))
    h = torch.zeros(1, model.decoder.lstm.hidden_size, device=device)
    c = torch.zeros(1, model.decoder.lstm.hidden_size, device=device)
    word = torch.tensor([vocab.stoi["<SOS>"]], device=device)

    caption = []
    alphas = []
    for _ in range(max_len):
        emb = model.decoder.embedding(word)
        context, alpha = model.decoder.attention(encoder_out, h)
        h, c = model.decoder.lstm(torch.cat([emb, context], dim=1), (h, c))
        word = model.decoder.fc(h).argmax(1)
        predicted_word = vocab.itos[word.item()]
        if predicted_word == "<EOS>":
            break
        caption.append(predicted_word)
        alphas.append(alpha.cpu().view(-1).numpy())
    return caption, alphas


def generate_caption(model, image, vocab, max_len=20, device=None):
    """Greedy decoding. Returns only the caption (list of words)."""
    return generate_caption_with_attention(model, image, vocab, max_len, device)[0]


def visualize_attention(image, caption, alphas, cols=5, save_path=None):
    """Show (and optionally save) the image with one attention overlay per generated word."""
    if not caption:
        print("The model produced an empty caption, nothing to visualise.")
        return

    img_np = image.permute(1, 2, 0).cpu().numpy()
    img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img_np = np.clip(img_np, 0, 1)
    height, width = img_np.shape[:2]

    rows = math.ceil(len(caption) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")

    for i, word in enumerate(caption):
        side = int(round(math.sqrt(alphas[i].shape[0])))
        att_map = alphas[i].reshape(side, side).astype(np.float32)
        att_resized = np.array(Image.fromarray(att_map).resize((width, height), resample=Image.BILINEAR))
        axes[i].imshow(img_np)
        axes[i].imshow(att_resized, alpha=0.5, cmap="jet")
        axes[i].set_title(word)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()
