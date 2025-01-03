from funasr import AutoModel
import torch
import mlflow
import yaml

wav_file = "/Users/kridtaphadsae-khow/Workspace/poc/โหนกระแส/chunks/บอสพอล/บอสพอล_9.wav"


voice_encoder = AutoModel(model="iic/emotion2vec_base")

classifier = mlflow.pytorch.load_model(
    "s3://staging-ai-mlflow/409241894704317319/c4254ad62fdf43d1a541acbef4e1a0c2/artifacts/best_model",
    map_location="cpu",
)
config_path = mlflow.artifacts.download_artifacts(
    "s3://staging-ai-mlflow/409241894704317319/c4254ad62fdf43d1a541acbef4e1a0c2/artifacts/training_args/thaiser.yaml"
)
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
    label2id = config["dataset"]["labels"]

id2label = {v: k for k, v in label2id.items()}

with torch.no_grad():
    feat = voice_encoder.generate(wav_file, granularity="frame")[0]["feats"]
    feat = torch.from_numpy(feat).unsqueeze(0)
    padding_mask = torch.zeros(feat.shape[0], feat.shape[1]).bool()
    logits = classifier(feat, padding_mask=padding_mask)
    probs = torch.softmax(logits, dim=-1)
    predicted_emotion = id2label[probs.argmax().item()]

    # Create probability dictionary for each emotion
    emotion_probs = {id2label[i]: prob.item() for i, prob in enumerate(probs[0])}

print("Predicted emotion:", predicted_emotion)
print("Emotion probabilities:", emotion_probs)
