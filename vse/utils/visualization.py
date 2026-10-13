"""Greedy caption generation and attention-map visualisation.

NOTE: Greedy decoding does not apply dropout (model.eval() is called).
This matches the original inference behaviour.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2


def generate_caption(model, image, vocab, max_len=20, device="cpu"):
    """Generate a caption for a single image using greedy decoding.

    Args:
        model: :class:`~vse.models.CombinedModel` instance.
        image: (C, H, W) tensor (un-batched).
        vocab: :class:`~vse.data.Vocabulary` instance.
        max_len: maximum caption length.
        device: torch device string.

    Returns:
        List of predicted word strings (excluding <SOS> and <EOS>).
    """
    model.eval()
    with torch.no_grad():
        encoder_out = model.encoder(image.unsqueeze(0).to(device))
        h = torch.zeros(1, model.decoder.lstm.hidden_size).to(device)
        c = torch.zeros(1, model.decoder.lstm.hidden_size).to(device)
        word = torch.tensor([vocab.stoi["<SOS>"]]).to(device)
        caption = []
        for _ in range(max_len):
            emb = model.decoder.embedding(word)
            context, _ = model.decoder.attention(encoder_out, h)
            lstm_input = torch.cat([emb, context], dim=1)
            h, c = model.decoder.lstm(lstm_input, (h, c))
            output = model.decoder.fc(h)
            predicted = output.argmax(1)
            word = predicted
            predicted_word = vocab.itos[predicted.item()]
            if predicted_word == "<EOS>":
                break
            caption.append(predicted_word)
        return caption


def generate_caption_with_attention(model, image, vocab, max_len=20, device="cpu"):
    """Generate a caption and per-word attention maps for a single image.

    Returns:
        caption: list of predicted words.
        alphas: list of numpy arrays (num_pixels,) for each word.
    """
    model.eval()
    with torch.no_grad():
        image = image.unsqueeze(0).to(device)  # (1, C, H, W)
        encoder_out = model.encoder(image)     # (1, num_pixels, embed_size)
        h = torch.zeros(1, model.decoder.lstm.hidden_size).to(device)
        c = torch.zeros(1, model.decoder.lstm.hidden_size).to(device)
        word = torch.tensor([vocab.stoi["<SOS>"]]).to(device)

        caption = []
        alphas = []

        for _ in range(max_len):
            emb = model.decoder.embedding(word)          # (1, embed_size)
            context, alpha = model.decoder.attention(encoder_out, h)
            lstm_input = torch.cat([emb, context], dim=1)
            h, c = model.decoder.lstm(lstm_input, (h, c))
            output = model.decoder.fc(h)
            predicted = output.argmax(1)
            word = predicted
            predicted_word = vocab.itos[predicted.item()]
            if predicted_word == "<EOS>":
                break
            caption.append(predicted_word)
            alphas.append(alpha.cpu().view(-1).numpy())  # flatten for resizing

    return caption, alphas


def visualize_attention(image, caption, alphas, save_path=None):
    """Overlay attention heatmaps on the image for each predicted word.

    Args:
        image: (C, H, W) tensor (ImageNet-normalised).
        caption: list of word strings.
        alphas: list of numpy arrays (one per word).
        save_path: if given, save the figure instead of showing it.
    """
    img_np = image.permute(1, 2, 0).cpu().numpy()  # (H, W, C)
    img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img_np = np.clip(img_np, 0, 1)

    num_words = len(caption)
    plt.figure(figsize=(3 * num_words, 3))

    for i, word in enumerate(caption):
        plt.subplot(1, num_words, i + 1)
        plt.imshow(img_np)
        att_map = alphas[i].squeeze()
        # Reshape attention map assuming square feature map
        H = W = int(np.sqrt(att_map.shape[0]))
        att_map = att_map.reshape(H, W)
        # Resize attention map to image size
        att_map_resized = cv2.resize(att_map, (img_np.shape[1], img_np.shape[0]))
        plt.imshow(att_map_resized, alpha=0.6, cmap="jet")
        plt.axis("off")
        plt.title(word)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
