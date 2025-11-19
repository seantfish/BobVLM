import torch
from torch.utils.data import Dataset
from transformers.image_utils import load_image
import json

class VLMDataset(Dataset):
    def __init__(self, data_dir, processor):
        """
        Args:
            data_dir (str): Directory containing the dataset
            processor: BobVLMProcessor instance
        """
        self.processor = processor
        with open(data_dir, 'r') as f:
            self.data = json.load(f)[5000*4:]
        # Load your dataset here
        # Example structure:
        # self.data = [
        #     {"image_path": "path/to/image.jpg", "role": "description"}
        # ]

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        item = self.data[idx]
        image = load_image(item['image_path'])
        inputs = self.processor(
            text=item["conversations"],
            images=image,
            return_tensors="pt",
        )
        inputs = prepare_labels(inputs)

        for k in inputs:
            if isinstance(inputs[k], torch.Tensor):
                inputs[k] = inputs[k].squeeze(0)
        return inputs
