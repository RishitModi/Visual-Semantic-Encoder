"""Tests for the Visual-Semantic-Encoder package.

All tests run on CPU with random tensors — no dataset downloads, no GPU.
They verify shapes, wiring, and checkpoint-key compatibility only.
"""

import torch
import pytest

from vse.models.encoder import EncoderCNN
from vse.models.attention import Attention
from vse.models.decoder import DecoderWithAttention
from vse.models.combined_model import CombinedModel
from vse.data.vocab import Vocabulary
from vse.data.dataset import CocoDataset, CaptionCollator
from vse.utils.visualization import generate_caption, generate_caption_with_attention
from vse.utils.config import Config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

EMBED = 256
HIDDEN = 512
ATTENTION_DIM = 256
VOCAB_SIZE = 50  # small dummy vocab
BATCH = 4
IMG_SIZE = 224
NUM_PIXELS = 49  # 7x7 grid from the 5-layer encoder
MAX_LEN = 20


@pytest.fixture
def vocab():
    """Build a tiny vocabulary from made-up sentences."""
    v = Vocabulary(freqthreshold=1)
    sentences = [
        "a cat on a mat",
        "a dog and a cat",
        "the big brown dog",
    ]
    v.buildvocabulary(sentences)
    return v


@pytest.fixture
def model():
    return CombinedModel(EMBED, HIDDEN, VOCAB_SIZE, attention_dim=ATTENTION_DIM)


# ---------------------------------------------------------------------------
# Encoder tests
# ---------------------------------------------------------------------------

class TestEncoder:
    def test_output_shape(self):
        enc = EncoderCNN(EMBED)
        x = torch.randn(BATCH, 3, IMG_SIZE, IMG_SIZE)
        out = enc(x)
        assert out.shape == (BATCH, NUM_PIXELS, EMBED)

    def test_different_batch_sizes(self):
        enc = EncoderCNN(EMBED)
        for b in [1, 2, 8]:
            x = torch.randn(b, 3, IMG_SIZE, IMG_SIZE)
            out = enc(x)
            assert out.shape == (b, NUM_PIXELS, EMBED)


# ---------------------------------------------------------------------------
# Attention tests
# ---------------------------------------------------------------------------

class TestAttention:
    def test_output_shape(self):
        att = Attention(EMBED, HIDDEN, ATTENTION_DIM)
        encoder_out = torch.randn(BATCH, NUM_PIXELS, EMBED)
        h = torch.randn(BATCH, HIDDEN)
        context, alpha = att(encoder_out, h)
        assert context.shape == (BATCH, EMBED)
        assert alpha.shape == (BATCH, NUM_PIXELS, 1)

    def test_weights_sum_to_one(self):
        att = Attention(EMBED, HIDDEN, ATTENTION_DIM)
        encoder_out = torch.randn(BATCH, NUM_PIXELS, EMBED)
        h = torch.randn(BATCH, HIDDEN)
        _, alpha = att(encoder_out, h)
        sums = alpha.squeeze(-1).sum(dim=1)  # (B,)
        assert torch.allclose(sums, torch.ones(BATCH), atol=1e-5)


# ---------------------------------------------------------------------------
# Decoder tests
# ---------------------------------------------------------------------------

class TestDecoder:
    def test_output_shape(self):
        """The decoder slices captions[:, :-1], so T-length input → T-2 outputs."""
        dec = DecoderWithAttention(EMBED, HIDDEN, VOCAB_SIZE, ATTENTION_DIM)
        encoder_out = torch.randn(BATCH, NUM_PIXELS, EMBED)
        T = 10
        captions = torch.randint(0, VOCAB_SIZE, (BATCH, T))
        # NOTE: The decoder internally does captions[:, :-1] giving T-1 tokens,
        # then produces T-2 output steps (one per embedding except the last
        # is dropped by the :-1 slice).  But wait — it slices captions[:, :-1]
        # to get embeddings of length T-1, then iterates T-1 times producing
        # T-1 outputs... Actually let me re-read: embeddings = self.embedding(
        # captions[:, :-1]) gives (B, T-1, embed), then loop runs T-1 times.
        # So output is (B, T-1, vocab_size).
        out = dec(encoder_out, captions)
        assert out.shape == (BATCH, T - 1, VOCAB_SIZE)


# ---------------------------------------------------------------------------
# CombinedModel tests
# ---------------------------------------------------------------------------

class TestCombinedModel:
    def test_forward_shape(self, model):
        images = torch.randn(BATCH, 3, IMG_SIZE, IMG_SIZE)
        T = 10
        captions = torch.randint(0, VOCAB_SIZE, (BATCH, T))
        # CombinedModel.forward passes captions directly to decoder, which
        # slices :-1 → output length = T-1
        out = model(images, captions)
        assert out.shape == (BATCH, T - 1, VOCAB_SIZE)

    def test_double_slicing_in_training(self, model):
        """Reproduce the exact double-slicing from train.py.

        train.py does: outputs = model(images, captions[:, :-1])
        Inside decoder.forward: embeddings = embedding(captions[:, :-1])
        So total slicing is twice: T → T-1 → T-2 steps of output.
        """
        images = torch.randn(BATCH, 3, IMG_SIZE, IMG_SIZE)
        T = 12
        captions = torch.randint(0, VOCAB_SIZE, (BATCH, T))
        # Simulating what train.py does:
        outputs = model(images, captions[:, :-1])
        # captions[:, :-1] has length T-1, decoder slices again → T-2
        assert outputs.shape == (BATCH, T - 2, VOCAB_SIZE)

    def test_backward_runs(self, model):
        images = torch.randn(BATCH, 3, IMG_SIZE, IMG_SIZE)
        captions = torch.randint(0, VOCAB_SIZE, (BATCH, 10))
        out = model(images, captions)
        loss = out.sum()
        loss.backward()
        # Check gradients exist
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None


# ---------------------------------------------------------------------------
# State dict key compatibility test
# ---------------------------------------------------------------------------

class TestStateDict:
    """Ensure the exact state_dict key prefixes match the trained checkpoint."""

    EXPECTED_PREFIXES = {
        "encoder.cnn.",
        "encoder.fc.",
        "decoder.attention.",
        "decoder.embedding.",
        "decoder.lstm.",
        "decoder.fc.",
    }

    def test_state_dict_key_prefixes(self, model):
        keys = set(model.state_dict().keys())
        # Every key must start with one of the expected prefixes
        for key in keys:
            prefix = ".".join(key.split(".")[:2]) + "."
            assert prefix in self.EXPECTED_PREFIXES, (
                f"Unexpected state_dict key prefix: {prefix!r} (from key {key!r})"
            )
        # Every expected prefix must appear at least once
        found_prefixes = {".".join(k.split(".")[:2]) + "." for k in keys}
        assert found_prefixes == self.EXPECTED_PREFIXES


# ---------------------------------------------------------------------------
# Vocabulary tests
# ---------------------------------------------------------------------------

class TestVocabulary:
    def test_special_tokens(self, vocab):
        assert vocab.stoi["<PAD>"] == 0
        assert vocab.stoi["<SOS>"] == 1
        assert vocab.stoi["<EOS>"] == 2
        assert vocab.stoi["<UNK>"] == 3

    def test_build_and_numericalize_round_trip(self, vocab):
        text = "a cat on a mat"
        indices = vocab.numericalize(text)
        words = [vocab.itos[i] for i in indices]
        assert words == ["a", "cat", "on", "a", "mat"]

    def test_unknown_token(self, vocab):
        indices = vocab.numericalize("xyzzy")
        assert indices == [vocab.stoi["<UNK>"]]

    def test_save_load_round_trip(self, vocab, tmp_path):
        path = tmp_path / "vocab.json"
        vocab.save(str(path))
        loaded = Vocabulary.load(str(path))
        assert len(loaded) == len(vocab)
        assert loaded.stoi == vocab.stoi
        assert loaded.itos == vocab.itos
        assert loaded.freq_threshold == vocab.freq_threshold


# ---------------------------------------------------------------------------
# CaptionCollator tests
# ---------------------------------------------------------------------------

class TestCaptionCollator:
    def test_padding(self):
        pad_idx = 0
        collator = CaptionCollator(pad_idx=pad_idx)
        # Fake batch: images are just tensors, captions have different lengths
        batch = [
            (torch.randn(3, IMG_SIZE, IMG_SIZE), torch.tensor([1, 4, 5, 2])),
            (torch.randn(3, IMG_SIZE, IMG_SIZE), torch.tensor([1, 6, 2])),
        ]
        images, captions = collator(batch)
        assert images.shape == (2, 3, IMG_SIZE, IMG_SIZE)
        assert captions.shape == (2, 4)  # padded to longest
        # Second caption should be padded with pad_idx
        assert captions[1, -1].item() == pad_idx


# ---------------------------------------------------------------------------
# Caption generation tests (untrained model, random outputs)
# ---------------------------------------------------------------------------

class TestCaptionGeneration:
    def test_generate_caption_returns_list(self, model, vocab):
        img = torch.randn(3, IMG_SIZE, IMG_SIZE)
        caption = generate_caption(model, img, vocab, max_len=MAX_LEN, device="cpu")
        assert isinstance(caption, list)
        assert len(caption) <= MAX_LEN
        # All items should be strings
        assert all(isinstance(w, str) for w in caption)

    def test_generate_caption_with_attention_returns_alphas(self, model, vocab):
        img = torch.randn(3, IMG_SIZE, IMG_SIZE)
        caption, alphas = generate_caption_with_attention(
            model, img, vocab, max_len=MAX_LEN, device="cpu"
        )
        assert isinstance(caption, list)
        assert isinstance(alphas, list)
        assert len(caption) == len(alphas)
        # Each alpha should have NUM_PIXELS elements
        if len(alphas) > 0:
            assert alphas[0].shape == (NUM_PIXELS,)
