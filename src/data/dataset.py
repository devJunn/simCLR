from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np
import albumentations as A
import torch
from typing import List


class CustomDataset(Dataset):
    def __init__(self, root: str, transforms: A.Compose):
        self.images = [image for image in Path(root).iterdir()]
        self.transforms = transforms
    
    def __getitem__(self, index: int) -> List[torch.Tensor]:
        image = Image.open(self.images[index])
        # image = np.array(image.convert("RGB").resize((224, 224)))
        image = np.array(image.convert("RGB").resize((224, 224)), dtype=np.float32)
        return [self.transforms(image=image)["image"] / 255. for _ in range(2)]

    def __len__(self):
        return len(self.images)