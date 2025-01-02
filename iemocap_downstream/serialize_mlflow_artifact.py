import os
import mlflow
import torch
import yaml


model = mlflow.pytorch.load_model(
    "s3://staging-ai-mlflow/409241894704317319/c4254ad62fdf43d1a541acbef4e1a0c2/artifacts/best_model",
    map_location="cpu",
)
model.eval()

config_path = mlflow.artifacts.download_artifacts(
    "s3://staging-ai-mlflow/409241894704317319/c4254ad62fdf43d1a541acbef4e1a0c2/artifacts/training_args/thaiser.yaml"
)
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
    label2id = config["dataset"]["labels"]

id2label = {v: k for k, v in label2id.items()}

feat = torch.rand(1, 768)
padding_mask = torch.zeros(1, 100).bool()

with torch.no_grad():
    logits = model(feat, padding_mask)
    pred = logits.argmax(dim=-1)
    predicted_emotion = id2label[pred.item()]

print("Logits:", logits)
print("Predicted emotion:", predicted_emotion)



