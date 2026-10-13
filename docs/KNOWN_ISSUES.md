# Known Issues and Limitations

This document lists known limitations of the original experiment and the
repository's code.  They are recorded here for transparency; none of them
change the fact that the system works end-to-end and produces captions with
meaningful attention maps.

---

## 1. Evaluation is in-sample

The evaluation script (`vse/evaluate.py`, formerly `inference/metric_eval.py`)
runs on the first 1,000 items of the **same training subset** used during
training — not on held-out images.  Each item provides a single reference
caption per image, and decoding is greedy (no beam search).  Only BLEU-1 and
BLEU-4 are computed (via NLTK `corpus_bleu`).

Any BLEU score obtained from this script should be understood as an
**in-sample** figure, not a measure of generalisation to unseen images.

### Author's reported results

The author reports **BLEU-1 = 0.39** and **BLEU-4 = 0.13** from the original
training run.  The full evaluation setup (exact data split, number of
references, decoding strategy) of that run is **not recorded in the
repository**, so these numbers are not reproducible from the repository alone
and should not be read as a held-out benchmark unless the author confirms it.

---

## 2. Training-loss plot only

The loss curve (`images/new_loss_curve.png`) shows **training loss** only.
There is no validation loss curve because the code does not compute validation
loss during training.

---

## 3. Training subset

The training script slices a contiguous range of the COCO 2017 annotation list
(controlled by `--num_train_pairs`).  This is a contiguous slice, not a random
sample of images.

---

## 4. Double caption slicing

In `vse/train.py`, the training loop passes `captions[:, :-1]` to
`CombinedModel.forward`.  Inside `DecoderWithAttention.forward`, the decoder
slices `captions[:, :-1]` **again**.  The net effect is that the input to the
LSTM drops the last **two** tokens of each caption (typically the `<EOS>` token
and the final content word).  The targets (`captions[:, 1:]`) are then
length-matched with a `min_len` clamp.

This is the behaviour the existing checkpoint was trained with.  Changing it
would break checkpoint compatibility.

---

## 5. Greedy decoding only

All inference and evaluation uses greedy decoding (argmax at each step).  Beam
search is not implemented.

---

## 6. From-scratch CNN encoder

The encoder (`EncoderCNN`) is a 5-layer strided convolutional network trained
from scratch — it does **not** use a pretrained backbone (e.g. ResNet,
EfficientNet).  This limits the quality of learned visual features compared to
transfer-learning baselines and results in simpler, less accurate captions.

---

## 7. Single reference per image in evaluation

Standard COCO captioning evaluation uses all five reference captions per image.
The repository's evaluation script uses only one reference per image (whichever
appears in the annotation-list entry for that data pair), which typically
deflates BLEU scores relative to multi-reference evaluation.
