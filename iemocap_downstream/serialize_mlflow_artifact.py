import os
import mlflow
import torch

os.environ["AWS_ACCESS_KEY_ID"] = "CHANGEME"
os.environ["AWS_SECRET_ACCESS_KEY"] = "CHANGEME"

model = mlflow.pytorch.load_model("s3://staging-ai-mlflow/938400390390876079/0f27f03086a943158e7c177c317423dc/artifacts/best_model", map_location="cpu")
model.eval()


feat = torch.rand(1,768)
padding_mask = torch.zeros(1, 100).bool()

with torch.no_grad():
    logits = model(feat, padding_mask)
    

