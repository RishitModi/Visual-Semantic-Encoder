from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from PIL import Image
from tqdm import tqdm

from .vocab import Vocabulary
from .visualization import generate_caption


def evaluate_bleu(model, vocab, val_pairs, transform, device, num_images=500):
    """BLEU-1 and BLEU-4 on validation images.

    val_pairs is a list of (image_path, caption) pairs from load_coco_pairs(..., "val2017").
    Every image is scored against all of its human reference captions.
    """
    model.eval()
    references_by_image = {}
    for path, caption in val_pairs:
        references_by_image.setdefault(path, []).append(Vocabulary.tokenizer(caption))

    references, hypotheses = [], []
    for path in tqdm(list(references_by_image)[:num_images], desc="BLEU evaluation"):
        image = transform(Image.open(path).convert("RGB"))
        hypotheses.append(generate_caption(model, image, vocab, device=device))
        references.append(references_by_image[path])

    bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25),
                        smoothing_function=SmoothingFunction().method1)
    return bleu1, bleu4
