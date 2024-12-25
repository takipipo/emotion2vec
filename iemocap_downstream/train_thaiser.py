import os

from pathlib import Path

import hydra 
from omegaconf import DictConfig

import torch
from torch import nn, optim

from data import load_ssl_features, train_valid_test_iemocap_dataloader
from model import BaseModel
from utils import train_one_epoch, validate_and_test
from torch.utils.data import DataLoader
from data import load_ssl_features, SpeechDataset
import logging
from tqdm import tqdm

logger = logging.getLogger('ThaiSER_Downstream')

def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        param = parameter.numel()
        print(f"{name}: {param}")
        total_params += param
    print(f"\nTotal number of parameters: {total_params}")

@hydra.main(config_path='config', config_name='thaiser.yaml')
def train_thaiser(cfg: DictConfig):
    # Get the absolute path to the project root directory
    project_dir = Path(__file__).parent.parent  # Go up one level from iemocap_downstream
    dataset_dir = project_dir / "dataset"  # Path to dataset directory
    
    # Define paths for each split
    train_path = str(dataset_dir / "train")
    val_path = str(dataset_dir / "val")
    test_path = str(dataset_dir / "test")
    
    # Verify all required files exist
    for base_path in [train_path, val_path, test_path]:
        for ext in ['.npy', '.lengths', '.emo']:
            if not os.path.exists(base_path + ext):
                raise FileNotFoundError(f"Required file not found: {base_path + ext}")
    
    logger.info("Loading datasets...")
    
    # Define emotion labels mapping
    label_dict = {'neutral': 0, 'happiness': 1, 'sadness': 2, 'anger': 3}  # Adjust based on your emotions
    
    # Load each dataset separately
    train_dataset = load_ssl_features(train_path, label_dict)
    val_dataset = load_ssl_features(val_path, label_dict)
    test_dataset = load_ssl_features(test_path, label_dict)
    
    # Create datasets
    train_set = SpeechDataset(
        feats=train_dataset['feats'],
        sizes=train_dataset['sizes'],
        offsets=train_dataset['offsets'],
        labels=train_dataset['labels']
    )
    
    val_set = SpeechDataset(
        feats=val_dataset['feats'],
        sizes=val_dataset['sizes'],
        offsets=val_dataset['offsets'],
        labels=val_dataset['labels']
    )
    
    test_set = SpeechDataset(
        feats=test_dataset['feats'],
        sizes=test_dataset['sizes'],
        offsets=test_dataset['offsets'],
        labels=test_dataset['labels']
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_set, 
        batch_size=cfg.dataset.batch_size, 
        collate_fn=train_set.collator,
        num_workers=4, 
        pin_memory=True, 
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_set, 
        batch_size=cfg.dataset.batch_size, 
        collate_fn=val_set.collator,
        num_workers=4, 
        pin_memory=True, 
        shuffle=False
    )
    
    test_loader = DataLoader(
        test_set, 
        batch_size=cfg.dataset.batch_size, 
        collate_fn=test_set.collator,
        num_workers=4, 
        pin_memory=True, 
        shuffle=False
    )
    
    torch.manual_seed(cfg.common.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    
    # Initialize model
    model = BaseModel(input_dim=768, output_dim=len(label_dict))
    model = model.to(device)

    # Setup optimizer and loss
    optimizer = optim.RMSprop(model.parameters(), lr=cfg.optimization.lr, momentum=0.9)
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=cfg.optimization.lr, max_lr=1e-3, step_size_up=10)
    criterion = nn.CrossEntropyLoss()

    best_val_wa = 0
    best_val_wa_epoch = 0
    
    # Create directory for saving models
    save_dir = os.path.join(str(Path.cwd()), "models")
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "best_model.pth")

    # Training loop
    logger.info("Starting training...")
    for epoch in tqdm(range(cfg.optimization.epoch), desc="Training"):
        train_loss = train_one_epoch(model, optimizer, criterion, train_loader, device)
        scheduler.step()
        
        # Validation step
        val_wa, val_ua, val_f1 = validate_and_test(model, val_loader, device, num_classes=len(label_dict))

        if val_wa > best_val_wa:
            best_val_wa = val_wa
            best_val_wa_epoch = epoch
            torch.save(model.state_dict(), model_path)
            logger.info(f"New best model saved with validation WA: {val_wa:.2f}%")

        # Print losses for every epoch
        logger.info(f"Epoch {epoch+1}, Training Loss: {train_loss/len(train_loader):.6f}, "
                   f"Validation WA: {val_wa:.2f}%; UA: {val_ua:.2f}%; F1: {val_f1:.2f}%")

    # Load best model and evaluate on test set
    logger.info("Loading best model for testing...")
    model.load_state_dict(torch.load(model_path))
    test_wa, test_ua, test_f1 = validate_and_test(model, test_loader, device, num_classes=len(label_dict))
    logger.info(f"Test Results - WA: {test_wa:.2f}%; UA: {test_ua:.2f}%; F1: {test_f1:.2f}%")

if __name__ == '__main__':
    train_thaiser()