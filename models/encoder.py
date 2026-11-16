import torch.nn as nn


class Bottleneck(nn.Module):
    """1x1 reduce -> 3x3 -> 1x1 expand, with a skip connection."""
    expansion = 4

    def __init__(self, in_channels, mid_channels, stride=1):
        super().__init__()
        out_channels = mid_channels * self.expansion
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + identity)


def make_stage(in_channels, mid_channels, num_blocks, stride):
    blocks = [Bottleneck(in_channels, mid_channels, stride)]
    in_channels = mid_channels * Bottleneck.expansion
    for _ in range(num_blocks - 1):
        blocks.append(Bottleneck(in_channels, mid_channels))
    return nn.Sequential(*blocks)


class EncoderCNN(nn.Module):
    """Custom CNN (ResNet-50-style, trained from scratch, fully trainable).

    (B, 3, 224, 224) -> (B, 2048, 7, 7) -> (B, 49, 2048) -> (B, 49, embed_size)
    """

    def __init__(self, embed_size, dropout_p=0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            make_stage(64, 64, 3, stride=1),
            make_stage(256, 128, 4, stride=2),
            make_stage(512, 256, 6, stride=2),
            make_stage(1024, 512, 3, stride=2),
        )
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(2048, embed_size)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.modules():
            if isinstance(m, Bottleneck):
                nn.init.zeros_(m.bn3.weight)

    def forward(self, images):
        features = self.cnn(images)
        B, C, H, W = features.shape
        features = features.permute(0, 2, 3, 1).reshape(B, H * W, C)
        return self.fc(self.dropout(features))
