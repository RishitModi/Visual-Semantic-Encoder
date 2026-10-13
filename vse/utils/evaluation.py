"""BLEU evaluation using NLTK corpus_bleu.

NOTE: The original inference/metric_eval.py evaluated on the first 1,000 items
of the same dataset used for training (not a held-out split), with a single
reference caption per image and greedy decoding.  See docs/KNOWN_ISSUES.md.
"""

import torch
from nltk.translate.bleu_score import corpus_bleu

from vse.utils.visualization import generate_caption


def evaluate_bleu(model, dataloader, vocab, device="cpu"):
    """Compute BLEU-1 and BLEU-4 over *dataloader*.

    Each batch yields ``(images, captions)`` where captions are
    numericalised tensors.  For each image the model generates a greedy
    caption, and the single ground-truth caption (excluding special
    tokens) is used as the reference.

    Returns:
        (bleu1, bleu4) floats.
    """
    model.eval()
    references = []
    hypotheses = []

    with torch.no_grad():
        for images, captions in dataloader:
            images = images.to(device)
            for i in range(images.size(0)):
                img = images[i]
                actual_caption = captions[i].cpu().numpy()
                predicted_caption = generate_caption(model, img, vocab, device=device)

                actual_tokens = [vocab.itos[idx]
                                 for idx in actual_caption if idx not in [0, 1, 2, 3]]
                predicted_tokens = predicted_caption

                references.append([actual_tokens])
                hypotheses.append(predicted_tokens)

    bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
    return bleu1, bleu4
