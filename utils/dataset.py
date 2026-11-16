import json
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def build_transform():
    """Preprocessing used for training and inference (ImageNet statistics, 224x224)."""
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_coco_pairs(dataset_dir, split="train2017"):
    """Return [(image_path, caption), ...] for a COCO split ('train2017' or 'val2017')."""
    dataset_dir = Path(dataset_dir)
    captions_path = dataset_dir / "annotations" / f"captions_{split}.json"
    if not captions_path.exists():
        raise FileNotFoundError(f"COCO captions file not found: {captions_path}")

    with captions_path.open("r", encoding="utf-8") as f:
        captions_json = json.load(f)

    images_dir = dataset_dir / split
    id_to_filename = {img["id"]: img["file_name"] for img in captions_json["images"]}
    pairs = [
        (str(images_dir / id_to_filename[ann["image_id"]]), ann["caption"])
        for ann in captions_json["annotations"]
        if ann["image_id"] in id_to_filename
    ]
    if not pairs:
        raise ValueError(f"No image-caption pairs found in dataset directory: {dataset_dir}")
    return pairs


class CocoDataset(Dataset):
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

        numericalized_caption = (
            [self.vocab.stoi["<SOS>"]]
            + self.vocab.numericalize(caption)
            + [self.vocab.stoi["<EOS>"]]
        )
        return image, torch.tensor(numericalized_caption)


def collate_fn(batch, vocab):
    images, captions = zip(*batch)
    images = torch.stack(images, 0)
    captions = pad_sequence(list(captions), batch_first=True, padding_value=vocab.stoi["<PAD>"])
    return images, captions
