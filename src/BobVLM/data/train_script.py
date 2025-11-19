import torch
from torch.utils.data import Dataset, DataLoader
from transformers.image_utils import load_image
import os
from tqdm import tqdm
import logging
from datetime import datetime
import json
from PIL import Image
import numpy as np
#from BobVLM.model import BobVLMProcessor,BobVLMConfig,BobVLM
import functools
from IPython.display import Javascript
from threading import Thread
#from peft import get_peft_config,get_peft_model,LoraConfig,TaskType, PeftModel


    
def load_adapter_weights(model, checkpoint_path):
    """
    Load adapter weights from a checkpoint file.
    
    Args:
        model: The model containing the adapter
        checkpoint_path (str): Path to the checkpoint file
    """
    try:
        # Load checkpoint with CPU/GPU handling
        if torch.cuda.is_available():
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        
        # Load adapter state dict
        model.adapter.load_state_dict(checkpoint['adapter_state_dict'])
        
        # Log information about the checkpoint
        print(f"Successfully loaded checkpoint from step {checkpoint['step']}")
        print(f"Checkpoint loss: {checkpoint['loss']:.4f}")
        
        # Verify adapter parameters are loaded
        trainable_params = sum(p.numel() for p in model.adapter.parameters())
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Adapter parameters loaded: {trainable_params:,}")
        print(f"Total parameters: {total_params:,}")
        
        return model
        
    except Exception as e:
        logger.error(f"Error loading checkpoint: {str(e)}")
        return False
        

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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

def prepare_labels(inputs):
    #Edit the labels such that the image token id is appended to the beginning of the original ids so they match for loss calc
    special_image_token_id = 64780 # the id corresponds to
    num_image_tokens = 257

    # Clone the input_ids to avoid modifying tensors in-place
    input_ids = inputs['input_ids'].clone()

    special_tokens_column = torch.full(
        (input_ids.size(0), num_image_tokens),
        float(special_image_token_id),
        device=input_ids.device,
        dtype=input_ids.dtype
    )

    # Create new tensor for labels
    inputs['labels'] = torch.cat([special_tokens_column, input_ids], dim=1).detach()

    return inputs

def create_dataset():
    # Initialize model and processor
    # initializing model for adapter training
    config = BobVLMConfig()
    model = BobVLM(config)
    model = load_adapter_weights(model, "/kaggle/input/checkpoints/pytorch/default/21/CLS_checkpoint_step_4000 (5).pt")

    # model = PeftModel.from_pretrained(
    #         model,
    #         "/kaggle/input/lora-checkpoints/pytorch/default/1/lora_checkpoint_step_6000",
    #         device_map={'': device},
    #     is_trainable=True
    #     )
    
    print('\n\n MODEL DESIGN\n',model)
    processor = BobVLMProcessor()

    # Create datasets and dataloaders
    train_dataset = VLMDataset("/kaggle/input/llava-instruct/llava_50k_1.json", processor)
    val_dataset = VLMDataset("/kaggle/input/sample/sample_data.json", processor)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Initialize trainer
    trainer = AdapterTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        learning_rate=1e-4,
        save_steps=1500,  # Save every 1000 steps
        log_steps=10,    # Log every 10 steps
    )

    # Start training
    trainer.train(num_epochs=1)

if __name__ == "__main__":
    main()
