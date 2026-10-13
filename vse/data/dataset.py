import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class CocoDataset(Dataset):
    """MS-COCO image-caption pair dataset.

    Each item is (image_tensor, caption_tensor) where the caption is
    numericalised with <SOS> / <EOS> framing.
    """

    def __init__(self, data_pairs, vocab, transform=None):
        self.data_pairs = data_pairs
        self.vocab = vocab
        self.transform = transform

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        imgpath, caption = self.data_pairs[idx]
        image = Image.open(imgpath).convert("RGB")
        if self.transform:
            image = self.transform(image)
        numericalized_caption = [self.vocab.stoi["<SOS>"]] + \
            self.vocab.numericalize(caption) + [self.vocab.stoi["<EOS>"]]
        return image, torch.tensor(numericalized_caption)


class CaptionCollator:
    """Collate function for DataLoader that pads captions to equal length.

    The pad index is passed explicitly so there is no dependency on a
    global ``vocab`` object.
    """

    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        images, captions = zip(*batch)
        images = torch.stack(images, 0)
        captions = [
            cap.clone() if isinstance(cap, torch.Tensor) else torch.tensor(cap)
            for cap in captions
        ]
        captions = pad_sequence(captions, batch_first=True, padding_value=self.pad_idx)
        return images, captions
