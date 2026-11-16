# Image Captioning Generator

This project is a PyTorch image captioning system that generates natural-language captions from input images using a custom ResNet-50-style CNN encoder, additive attention, and an LSTM-based decoder. The implementation is intentionally compact and research-oriented: it reads COCO-style caption annotations, builds a vocabulary from the training captions, trains an encoder-decoder model with teacher forcing, and can generate captions and attention maps for individual images.

The repository is structured as a small educational / research codebase rather than a production web app or packaged framework. The implementation and training flow are defined directly in the Python source files and can be executed from the project root.

## Project overview

The actual model pipeline in this repository is:

Image
→ image preprocessing
→ custom ResNet-50-style CNN encoder
→ spatial feature embedding
→ additive attention
→ LSTM decoder
→ vocabulary projection
→ generated caption

The architecture is implemented through the modules in the `models/` directory, while dataset loading, vocabulary construction, checkpoint handling, evaluation, and visualization utilities live under `utils/` and `inference/`.

## Highlights

- Custom ResNet-50-style CNN encoder for spatial feature extraction
- Additive soft attention between encoder features and decoder hidden state
- LSTMCell-based decoder for autoregressive caption generation
- Vocabulary tokenization and numericalization from COCO captions
- Teacher-forcing training with shifted target captions
- Checkpoints that also carry the vocabulary, plus a `vocab.txt` written during training
- Resumable training (`--resume`)
- COCO 2017 dataset support with configurable dataset path
- BLEU-based evaluation via NLTK
- Attention-map visualization for generated captions
- CPU-first training/inference with optional CUDA when available

## Example outputs from the project

> The example images below were generated with an earlier encoder version, so captions from the current model may differ.

The repository includes example result images under the `images/` directory. These are useful for understanding the type of output produced by the captioning pipeline: a model-generated caption is compared against the visual content of the image, and the attention maps show which image regions were emphasized while each word was being produced.

### Generated caption examples

<p align="center">
    <img src="images/output_image1.png" width="260" alt="Example generated caption 1">
    <img src="images/output_image2.png" width="260" alt="Example generated caption 2">
    <img src="images/output_image3.png" width="260" alt="Example generated caption 3">
</p>

<p align="center">
    <img src="images/output_image4.png" width="260" alt="Example generated caption 4">
    <img src="images/output_image5.png" width="260" alt="Example generated caption 5">
</p>

These outputs show the model producing short, image-grounded descriptions such as identifying a person, vehicle, object, or scene. The model is not designed to generate highly polished captions every time; it is a compact research implementation that favors clear visual grounding over production-level language quality.

### Attention-map examples

<p align="center">
    <img src="images/heatmap_image-2.png" width="420" alt="Attention map showing model focus on image regions">
    <img src="images/heatmap_other.png" width="420" alt="Second attention map showing model focus on image regions">
</p>

The attention maps highlight the spatial regions the model focused on while predicting particular words. In these visualizations, brighter areas indicate stronger attention. This makes the decoder easier to inspect: instead of being a black box, it can be interpreted as choosing the image regions most relevant to a word such as an object, person, or scene element.

## Repository structure

```text
Visual-Semantic-Encoder/
├── README.md
├── requirements.txt
├── .gitignore
├── train.py
├── data/
│   └── MSCOCO.txt
├── images/
│   ├── heatmap_image-2.png
│   ├── heatmap_other.png
│   ├── loss_curve.png
│   ├── new_loss_curve.png
│   ├── output_image1.png
│   ├── output_image2.png
│   ├── output_image3.png
│   ├── output_image4.png
│   └── output_image5.png
├── checkpoints_custom/
│   ├── checkpoint_best.pth
│   ├── checkpoint_img_caption.pth
│   └── metrics.json
├── scripts/
│   └── repack_pth.py
├── inference/
│   ├── caption_pred.py
│   └── metric_eval.py
├── models/
│   ├── attention.py
│   ├── combined_model.py
│   ├── decoder.py
│   └── encoder.py
└── utils/
    ├── checkpoint.py
    ├── dataset.py
    ├── evaluation.py
    ├── visualization.py
    └── vocab.py
```

## Model architecture

The model is an encoder-decoder captioning network: a custom ResNet-50-style CNN image encoder, additive attention, and an LSTM decoder. It is not a transformer.

### Encoder

Implemented in `models/encoder.py`.

The encoder is a custom ResNet-50-style bottleneck CNN with 3-4-6-3 blocks. It is written from scratch, randomly initialised with Kaiming initialisation, and trained end to end. It has no pretrained weights and does not download ImageNet weights.

- Input: `(B, 3, 224, 224)`
- ResNet-50-style bottleneck stages -> feature map `(B, 2048, 7, 7)`
- after permute and reshape: `(B, 49, 2048)` (49 regions)
- Dropout(0.3)
- Linear(2048, 256) -> `(B, 49, 256)`

`EncoderCNN(embed_size)` is defined in `models/encoder.py` and is fully trainable.

### Attention mechanism

Implemented in `models/attention.py`.

This is additive / soft attention:

- `encoder_att`: linear projection from encoder dimension to attention dimension
- `decoder_att`: linear projection from decoder hidden state to attention dimension
- `tanh(encoder_att + decoder_att)`
- `full_att` produces a scalar attention score per spatial location
- `softmax` across all spatial positions
- weighted context vector: `(encoder_out * alpha).sum(dim=1)`

The attention module returns:

- `context`: `(B, encoder_dim)`
- `alpha`: attention weights of shape `(B, num_pixels, 1)`

### Decoder

Implemented in `models/decoder.py`.

The decoder uses:

- `nn.Embedding(vocab_size, embed_size)`
- `nn.LSTMCell(embed_size + embed_size, hidden_size)`
- `nn.Linear(hidden_size, vocab_size)`
- dropout before the final vocab projection

The decoder receives:

- encoder output: `(B, num_pixels, embed_size)`
- caption tokens: `(B, T)`

The decoder shifts the captions itself: it reads `captions[:, :-1]` as input, and its outputs are compared to `captions[:, 1:]` using teacher forcing.

### Combined model

Implemented in `models/combined_model.py`.

The model wrapper simply connects:

`EncoderCNN` (ResNet-50) -> `DecoderWithAttention`

with the forward pass:

```python
encoder_out = self.encoder(images)
outputs = self.decoder(encoder_out, captions)
```

The output shape is `(B, T-1, vocab_size)`.

## Data flow

The training data pipeline in this repository is:

1. Read COCO caption JSON file from the dataset annotations directory.
2. Map each `image_id` to the corresponding image filename.
3. Build a list of `(image_path, caption)` pairs.
4. Tokenize captions using the vocabulary logic.
5. Convert tokens into integer IDs.
6. Add `<SOS>` and `<EOS>` around each caption.
7. Prepare a PyTorch `Dataset` that returns image tensor + caption tensor.
8. Pad captions in a batch with `<PAD>` tokens.
9. Feed the image batch and shifted caption batch into the model.
10. Compute cross-entropy loss against the shifted target captions.

## Training pipeline

The actual training logic lives in `train.py` and uses the following default configuration:

- `--dataset-dir` default: `data/coco2017`
- `--epochs` default: `10`
- `--batch-size` default: `64`
- `--subset-size` default: `100000` (`0` uses all captions)
- `--lr` default: `4e-4` (AdamW with a cosine schedule)
- `--val-subset-size` default: `5000` (`0` skips validation)
- `--weight-decay` default: `1e-4`
- `--seed` default: `42`
- `--no-amp` default: disabled
- `--freq-threshold` default: `5`
- `--num-workers` default: `0` on Windows, `2` elsewhere
- `--checkpoint-dir` default: `checkpoints_custom`
- `--resume` optional: continue from an existing checkpoint

The script performs the following steps:

1. Reads the COCO `captions_train2017.json` file.
2. Builds a vocabulary from all training captions and saves it to `checkpoints_custom/vocab.txt`.
3. Creates `CocoDataset` instances using the image/caption pairs.
4. Selects a subset of samples using `Subset(dataset, range(subset_size))`.
5. Constructs a `DataLoader` with `collate_fn=partial(collate_fn, vocab=vocab)`.
6. Instantiates the custom CNN model and the optimizer.
7. Trains using `nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"])` with gradient clipping.
8. Runs validation after each epoch.
9. Saves `checkpoint_img_caption.pth`, `checkpoint_best.pth`, and `metrics.json` in the checkpoint directory.
10. Writes a loss curve with train and validation lines.

The optimizer updates all trainable model parameters:

```python
torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
```

The model runs on:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

This means the project is written to support CPU execution and optional CUDA execution when available, but the repository does not assume CUDA is present.

## Vocabulary and tokenization

The vocabulary implementation is in `utils/vocab.py`.

Special tokens are defined as:

- `0: "<PAD>"`
- `1: "<SOS>"`
- `2: "<EOS>"`
- `3: "<UNK>"`

The tokenizer lowercases text and extracts words using a regex-based word split:

```python
re.findall(r"\b\w+\b", text)
```

The vocabulary is built by counting tokens across the training captions and keeping words whose frequency is greater than or equal to the configured threshold.

The numericalization step maps each word to its integer ID, falling back to `<UNK>` when a word is not in the vocabulary.

## Data preprocessing

Training and inference apply the following transforms, defined in `build_transform()` in `utils/dataset.py`:

```python
transforms.Resize((224, 224))
transforms.ToTensor()
transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)
```

This matches the standard ImageNet-like normalization used for the encoder input. Images are loaded as RGB using Pillow before transformation.

## Dataset assumptions

The repository expects a COCO 2017 dataset root with a structure like:

```text
<dataset_dir>/
├── train2017/
│   ├── 000000000001.jpg
│   ├── 000000000002.jpg
│   └── ...
└── annotations/
    └── captions_train2017.json
```

For BLEU evaluation the dataset also needs `val2017/` and `annotations/captions_val2017.json`.

The layout is also described in `data/MSCOCO.txt`.

Important notes:

- The dataset itself is not included in the repository.
- The code expects the COCO data to exist locally.
- There is no automatic download step in the project.
- The training script accepts a dataset directory via `--dataset-dir`.

## Checkpointing

Checkpoint behavior is implemented in `utils/checkpoint.py`.

`save_checkpoint` stores:

- `epoch`
- `model_state_dict`
- `optimizer_state_dict`
- `loss`
- `vocab_itos` (the vocabulary, so the checkpoint is self-contained)

`train.py` also writes the vocabulary to `checkpoints_custom/vocab.txt` (one `index:word` line per entry). The default checkpoint path is:

```text
checkpoints_custom/checkpoint_img_caption.pth
```

The checkpoint directory is created automatically if it does not exist. Checkpoints are not committed to the repository (see `.gitignore`).
The training script refuses to overwrite an existing checkpoint unless `--resume` is used. `metrics.json` is written next to the checkpoints.

For inference, `load_for_inference(checkpoint_path, vocab_path=None, device="cpu")` rebuilds the model and vocabulary. Model sizes are read from the saved weights, and `vocab_path` is only needed for checkpoints that do not contain a vocabulary. To resume training, `load_checkpoint(model, optimizer, path)` restores the weights and optimizer state.

## Inference and caption generation

The caption generation utility is in `utils/visualization.py`.

The function `generate_caption` performs greedy decoding:

- start with `<SOS>`
- encode the image once
- initialize hidden state and cell state to zeros
- at each step:
  - embed the current word
  - compute attention context with current hidden state
  - concatenate embedding and context
  - pass to `LSTMCell`
  - project hidden state to vocabulary logits
  - choose the argmax token
  - stop when `<EOS>` is reached or max length is reached

The function `generate_caption_with_attention` returns both the predicted caption and the attention weights recorded during generation.

## BLEU evaluation

The evaluation utility is implemented in `utils/evaluation.py`.

It computes:

- BLEU-1
- BLEU-4

using `nltk.translate.bleu_score.corpus_bleu` (BLEU-4 uses smoothing).

Evaluation runs on COCO `val2017` images the model was not trained on, and each image is scored against all of its human reference captions. The script `inference/metric_eval.py` wraps this with a CLI that expects a local COCO dataset and a checkpoint.
Compare checkpoints with the same `--num-images` so the numbers are comparable.

## Attention visualization

The attention visualization logic is in `utils/visualization.py`.

`visualize_attention(image, caption, alphas)` does the following:

1. Converts the image tensor back to a NumPy image.
2. Reverses the normalization to show the original image colors approximately.
3. Reshapes each attention map to a 2D spatial grid.
4. Resizes it to the original image dimensions.
5. Plots the image and attention overlay for each generated word.

This provides a qualitative map of the attention locations used during caption generation.

## Inference scripts

The repository includes several execution-oriented scripts. Run them from the project root with `python -m` (for the scripts inside `inference/`) so that the `models/` and `utils/` packages are found.

### `train.py`

Main training entry point.

```bash
python train.py --dataset-dir "D:\path\to\coco2017" --epochs 10 --batch-size 64 --subset-size 100000 --checkpoint-dir checkpoints_custom
```

### `inference/caption_pred.py`

Generates a caption for a single image from a checkpoint and shows the attention map for each word.

```bash
python -m inference.caption_pred --image "D:\path\to\image.jpg" --checkpoint checkpoints_custom/checkpoint_img_caption.pth
```

Useful options: `--vocab checkpoints_custom/vocab.txt` (only for checkpoints without a built-in vocabulary), `--no-plot` (print the caption only), `--save out.png` (save the attention figure).

### `inference/metric_eval.py`

Evaluates a trained model with BLEU on COCO `val2017`.

```bash
python -m inference.metric_eval --dataset-dir "D:\path\to\coco2017" --checkpoint checkpoints_custom/checkpoint_img_caption.pth --num-images 500
```

## Dependencies

The project relies on the packages listed in `requirements.txt`:

- `numpy==1.25.0`
- `matplotlib==3.8.0`
- `Pillow==10.1.0`
- `tqdm==4.66.1`
- `torch==2.1.0`
- `torchvision==0.16.0`
- `nltk==3.8.1`

This is the actual dependency set used by the project source files.

## Environment notes

The project is designed to run with Python 3.11 and the installed PyTorch/TorchVision versions in the working environment. CUDA is optional and selected automatically when available (a GPU is strongly recommended for training). The repository does not ship a custom training runtime or a Dockerfile.

## Quick start

1. Create a virtual environment:

```bash
python -m inference.metric_eval --dataset-dir path/to/coco2017 --checkpoint checkpoints_custom/checkpoint_img_caption.pth --num-images 500
```

This needs `val2017/` and `annotations/captions_val2017.json` (see [Training your own model](#training-your-own-model) for the layout).

---

## Training your own model

### 1. Get the dataset

Download **MS COCO 2017** from [cocodataset.org](https://cocodataset.org/#download): the *2017 Train images*, *2017 Val images* and *2017 Train/Val annotations*. Arrange them like this:

```text
coco2017/
├── train2017/                       # about 118,000 training images
├── val2017/                         # about 5,000 validation images (for BLEU)
└── annotations/
    ├── captions_train2017.json
    └── captions_val2017.json
```

Each image has about five human-written captions.

### 2. Train

```bash
python train.py --dataset-dir path/to/coco2017 --epochs 10 --batch-size 64 --subset-size 100000 --checkpoint-dir checkpoints_custom
```

| Option | Default | Meaning |
|---|---|---|
| `--dataset-dir` | `data/coco2017` | COCO root folder |
| `--epochs` | `10` | Passes over the selected captions |
| `--batch-size` | `64` | Samples per training step |
| `--subset-size` | `100000` | Number of training captions to use. `0` = use all (about 590,000) |
| `--lr` | `4e-4` | AdamW learning rate with a cosine schedule |
| `--val-subset-size` | `5000` | Number of validation captions. `0` = skip validation |
| `--weight-decay` | `1e-4` | AdamW weight decay |
| `--seed` | `42` | Random seed |
| `--no-amp` | disabled | Disable automatic mixed precision |
| `--freq-threshold` | `5` | Minimum word count to enter the vocabulary |
| `--num-workers` | `0` on Windows, `2` elsewhere | Data-loading worker processes |
| `--checkpoint-dir` | `checkpoints_custom` | Where the outputs are written |
| `--resume FILE` | none | Continue from a saved checkpoint |

What happens:

1. All training captions are read and the vocabulary is built and saved to `checkpoints_custom/vocab.txt`.
2. The first captions (up to `--subset-size`) are used for training.
3. After **every epoch**, validation runs on up to `--val-subset-size` captions.
4. The checkpoint is saved to `checkpoints_custom/checkpoint_img_caption.pth`, and the best validation checkpoint is saved to `checkpoints_custom/checkpoint_best.pth`.
5. `metrics.json` and a loss curve with train and validation lines are written to `checkpoints_custom/`.

Tips:

- **Use a GPU.** A free [Kaggle](https://www.kaggle.com) notebook GPU works well: add a COCO 2017 dataset, switch on the GPU and Internet, and point `--dataset-dir` at the dataset folder.
- Do a **quick test first** with `--epochs 1 --subset-size 2000` to check that everything runs and to measure the time per epoch.
- More captions and more epochs give the from-scratch CNN more training data and updates.
- The checkpoint contains the trainable CNN and optimizer state.

---

## Checkpoint and vocabulary files

A checkpoint (`.pth`) is a PyTorch file that stores:

| Key | Content |
|---|---|
| `epoch` | Index of the last finished epoch |
| `model_state_dict` | All model weights |
| `optimizer_state_dict` | Optimizer state, needed only to resume training |
| `scheduler_state_dict` | Cosine learning-rate scheduler state |
| `loss` | Average training loss of that epoch |
| `val_loss` | Validation loss of that epoch |
| `vocab_itos` | The vocabulary (index to word), so the file is self-contained |

`vocab.txt` holds the same vocabulary as text, one `index:word` pair per line:

```text
0:<PAD>
1:<SOS>
2:<EOS>
3:<UNK>
4:a
5:man
...
```

For inference, `utils.checkpoint.load_for_inference(...)` rebuilds the model from the file. It reads the layer sizes from the saved weights themselves, so you do not have to pass the model dimensions.

Good to know:

- A `.pth` file is secretly a **zip archive**. If a download or unzip tool *extracts* it, you end up with a folder named `checkpoint_img_caption.pth` that contains `data.pkl`, `data/` and so on, and loading fails with `Permission denied`. Download the file again, or rebuild it with `python scripts/repack_pth.py <unpacked_folder> <new_file.pth>`.
- The vocabulary and the checkpoint **must come from the same training run**.
- Checkpoints from the older small-CNN version of this project do not fit the current model.

---

## API reference

The server (`app/server.py`) exposes two endpoints and also serves the web page from the `web/` folder.

### `GET /api/health`

Returns `{"status": "ok"}` once the server is running. The web page calls it on load to wake up a sleeping host.

### `POST /api/caption`

Upload an image as multipart form data in a field named `file`.

```bash
python train.py --dataset-dir "D:\data\coco2017" --epochs 10 --batch-size 64 --subset-size 100000 --checkpoint-dir checkpoints_custom
```

6. Generate a caption for an image:

```bash
python -m inference.caption_pred --image "D:\path\to\image.jpg" --checkpoint checkpoints_custom/checkpoint_img_caption.pth
```

## Example outputs

The repository includes example result images under `images/`. They show the project’s output format and the expected visualization style, but they are not a substitute for a trained checkpoint or a local COCO dataset.

## Known project limitations

- The project expects a local COCO dataset and does not include training data in the repository.
- There is no packaged trained model checkpoint in the repo.
- There is no web UI, API, or notebook-based demo packaged as a service.
- Training and evaluation are script-based rather than abstracted into a CLI framework.
- The implementation is intentionally minimal and focused on core research functionality rather than production polish.
- The caption generation process is greedy and uses the most probable next token at each step (no beam search).
- The from-scratch CNN needs enough data and epochs to train well.
- It has not yet been compared on BLEU with the earlier encoder version.

## Architecture diagram

```mermaid
flowchart LR
    A[Input Image] --> B[Resize + Normalize]
    B --> C[Custom CNN Encoder - ResNet-50-style]
    C --> D[Spatial Feature Map]
    D --> E[Additive Attention]
    E --> F[Context Vector]
    F --> G[LSTMCell Decoder]
    G --> H[Vocabulary Projection]
    H --> I[Next Token]
    I --> G
    I --> J[Generated Caption]
```

## Summary

This repository implements a compact, training-capable image captioning system in PyTorch. It uses a custom ResNet-50-style CNN encoder, additive soft attention, and an LSTM-based decoder to generate captions from COCO image-caption pairs. The code is designed to be understandable and practical for study, experimentation, and further refinement, rather than for deployment as a commercial-grade product.

## Contributors

<!-- List the team members here. -->
