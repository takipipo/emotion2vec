import os

from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
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
import mlflow
import mlflow.pytorch

os.environ["MLFLOW_TRACKING_USERNAME"] = "admin"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "password"

logger = logging.getLogger("ThaiSER_Downstream")



def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        param = parameter.numel()
        print(f"{name}: {param}")
        total_params += param
    print(f"\nTotal number of parameters: {total_params}")


@hydra.main(config_path="config", config_name="thaiser.yaml")
def train_thaiser(cfg: DictConfig):

    feat_path = os.path.join(get_original_cwd(), cfg.dataset.feat_path)

    # Verify all required files exist
    for split in ["train", "val", "test"]:
        for ext in [".npy", ".lengths", ".emo"]:
            if not os.path.exists(os.path.join(feat_path, split + ext)):
                raise FileNotFoundError(f"Required file not found: {split + ext}")

    logger.info("Loading datasets...")

    # Define emotion labels mapping from config
    label_dict = {k: v for k, v in cfg.dataset.labels.items()}

    # Load each dataset separately
    # load_ssl_features จะไปแอบ join .emo, .npy, .lengths
    train_path = os.path.join(feat_path, "train")
    val_path = os.path.join(feat_path, "val")
    test_path = os.path.join(feat_path, "test")

    train_dataset = load_ssl_features(train_path, label_dict)
    val_dataset = load_ssl_features(val_path, label_dict)
    test_dataset = load_ssl_features(test_path, label_dict)

    # Create datasets
    train_set = SpeechDataset(
        feats=train_dataset["feats"],
        sizes=train_dataset["sizes"],
        offsets=train_dataset["offsets"],
        labels=train_dataset["labels"],
    )

    val_set = SpeechDataset(
        feats=val_dataset["feats"],
        sizes=val_dataset["sizes"],
        offsets=val_dataset["offsets"],
        labels=val_dataset["labels"],
    )

    test_set = SpeechDataset(
        feats=test_dataset["feats"],
        sizes=test_dataset["sizes"],
        offsets=test_dataset["offsets"],
        labels=test_dataset["labels"],
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.dataset.batch_size,
        collate_fn=train_set.collator,
        num_workers=4,
        pin_memory=True,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=cfg.dataset.batch_size,
        collate_fn=val_set.collator,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=cfg.dataset.batch_size,
        collate_fn=test_set.collator,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
    )

    torch.manual_seed(cfg.common.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    # Initialize model
    model = BaseModel(input_dim=768, output_dim=len(label_dict))
    model = model.to(device)

    # Setup optimizer and loss
    optimizer = optim.RMSprop(model.parameters(), lr=cfg.optimization.lr, momentum=0.9)
    scheduler = optim.lr_scheduler.CyclicLR(
        optimizer, base_lr=cfg.optimization.lr, max_lr=1e-3, step_size_up=10
    )
    criterion = nn.CrossEntropyLoss()

    best_val_wa = 0
    best_val_wa_epoch = 0

    # Create directory for saving models
    save_dir = os.path.join(str(Path.cwd()), "models")
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "best_model.pth")

    # Initialize MLflow
    mlflow.set_tracking_uri("http://localhost:8000")
    mlflow.set_experiment("ThaiSER_Emotion_Recognition")

    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(
            {
                "batch_size": cfg.dataset.batch_size,
                "learning_rate": cfg.optimization.lr,
                "epochs": cfg.optimization.epoch,
                "seed": cfg.common.seed,
                "num_classes": len(label_dict),
            }
        )

        # Training loop
        logger.info("Starting training...")
        for epoch in tqdm(range(cfg.optimization.epoch), desc="Training"):
            train_loss = train_one_epoch(
                model, optimizer, criterion, train_loader, device
            )
            scheduler.step()

            # Validation step
            val_wa, val_ua, val_f1 = validate_and_test(
                model, val_loader, device, num_classes=len(label_dict)
            )

            # Log metrics
            mlflow.log_metrics(
                {
                    "train_loss": train_loss / len(train_loader),
                    "val_weighted_accuracy": val_wa,
                    "val_unweighted_accuracy": val_ua,
                    "val_f1": val_f1,
                },
                step=epoch,
            )

            if val_wa > best_val_wa:
                best_val_wa = val_wa
                best_val_wa_epoch = epoch
                torch.save(model.state_dict(), model_path)
                logger.info(f"New best model saved with validation WA: {val_wa:.2f}%")
                # Log best model
                mlflow.pytorch.log_model(model, "best_model")

            # Print losses for every epoch
            logger.info(
                f"Epoch {epoch+1}, Training Loss: {train_loss/len(train_loader):.6f}, "
                f"Validation WA: {val_wa:.2f}%; UA: {val_ua:.2f}%; F1: {val_f1:.2f}%"
            )

        # Test evaluation
        test_wa, test_ua, test_f1 = validate_and_test(
            model, test_loader, device, num_classes=len(label_dict)
        )

        # Log final test metrics
        mlflow.log_metrics(
            {
                "test_weighted_accuracy": test_wa,
                "test_unweighted_accuracy": test_ua,
                "test_f1": test_f1,
            }
        )

        logger.info(
            f"Test Results - WA: {test_wa:.2f}%; UA: {test_ua:.2f}%; F1: {test_f1:.2f}%"
        )


if __name__ == "__main__":
    train_thaiser()
