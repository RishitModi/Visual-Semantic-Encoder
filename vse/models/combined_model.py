import torch.nn as nn

from vse.models.encoder import EncoderCNN
from vse.models.decoder import DecoderWithAttention


class CombinedModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, attention_dim=256):
        super(CombinedModel, self).__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderWithAttention(embed_size, hidden_size, vocab_size, attention_dim)

    def forward(self, images, captions):
        encoder_out = self.encoder(images)  # (B, num_pixels, embed)
        outputs = self.decoder(encoder_out, captions)  # (B, T-1, vocab_size)
        return outputs
