# Training Guide for ThaiSER Dataset

This guide explains how to prepare and train a speech emotion recognition model on the ThaiSER dataset using Emotion2Vec features.

## Dataset Structure

ThaiSER dataset should be organized as follows:

```
dataset/
├── train/
│   ├── angry_001.wav
│   ├── sad_001.wav
│   └── ...
├── val/
│   ├── angry_101.wav
│   ├── sad_101.wav
│   └── ...
└── test/
    ├── angry_201.wav
    ├── sad_201.wav
    └── ...
```

## Step 1: Prepare CSV Files

Create CSV files for each split containing paths and emotions:

```python
# Example: create_csv.py
import pandas as pd
import os

def create_split_csv(split_dir, output_file):
    rows = []
    for file in os.listdir(split_dir):
        if file.endswith('.wav'):
            emotion = file.split('_')[0]  # Assuming filename format: emotion_id.wav
            path = os.path.join('./dataset', split_dir, file)
            rows.append({'PATH': path, 'EMOTION': emotion})
    
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)

# Create CSVs for each split
create_split_csv('train', 'train.csv')
create_split_csv('val', 'val.csv')
create_split_csv('test', 'test.csv')
```

## Step 2: Extract Features

> **Important**: Before extracting features, you need to:
> 1. Download the Emotion2Vec pre-trained checkpoint from the [main repository](../README.md#extract-features-from-your-dataset)
> 2. Place the checkpoint file in the `./upstream` folder of your project

Extract Emotion2Vec features for each split:

```bash
# Extract features for training set
python scripts/prepare_thaiser.py \
    --split train \
    --df_path train.csv \
    --out_dir features/

# Extract features for validation set
python scripts/prepare_thaiser.py \
    --split val \
    --df_path val.csv \
    --out_dir features/

# Extract features for test set
python scripts/prepare_thaiser.py \
    --split test \
    --df_path test.csv \
    --out_dir features/
```

This will create the following files in your features directory:
```
features/
├── train.npy       # Feature vectors
├── train.lengths   # Length of each sample
├── train.emo      # Emotion labels
├── val.npy
├── val.lengths
├── val.emo
├── test.npy
├── test.lengths
└── test.emo
```

## Step 3: Configure Training

Create or modify `config/thaiser.yaml`:

```yaml
dataset:
  feat_path: "features"  # Path to extracted features
  batch_size: 32
  labels:
    angry: 0
    happy: 1
    sad: 2
    neutral: 3
    fear: 4  # Adjust based on ThaiSER emotions

optimization:
  lr: 0.001
  epoch: 100

common:
  seed: 42
```

## Step 4: Training

Run the training script:

```bash
python train_thaiser.py
```

The script will:
1. Load the features and labels
2. Initialize the model
3. Train for the specified number of epochs
4. Save the best model based on validation performance
5. Evaluate on the test set

## Training Output

The training process will create:

```
outputs/
├── models/
│   └── best_model.pth    # Best model weights
└── logs/
    └── training.log      # Training progress log
```

## Monitoring Training

The training log will show:
- Training loss per epoch
- Validation metrics (WA, UA, F1)
- Best model saves
- Final test set performance

Example log output:
```
Epoch 1, Training Loss: 1.234567, Validation WA: 45.67%; UA: 43.21%; F1: 44.89%
New best model saved with validation WA: 45.67%
...
Test Results - WA: 47.89%; UA: 46.54%; F1: 47.12%
```

## Performance Metrics

The model reports three metrics:
- WA (Weighted Accuracy): Accuracy weighted by class distribution
- UA (Unweighted Accuracy): Average of per-class accuracies
- F1: Macro-averaged F1 score across all emotions

## Troubleshooting

Common issues and solutions:

1. CUDA out of memory
   - Reduce batch_size in config
   - Use shorter audio segments

2. Poor performance
   - Check class balance in training data
   - Adjust learning rate
   - Increase training epochs

3. Feature extraction fails
   - Verify audio file format (should be WAV)
   - Check sample rate (should be 16kHz)
   - Ensure mono channel audio

## Additional Notes

- Audio preprocessing:
  - Files are automatically converted to mono if stereo
  - Sample rate is resampled to 16kHz if needed
  - Silence removal is not performed by default

- Model checkpointing:
  - Best model is saved based on validation WA
  - Previous best model is overwritten
  - Final evaluation uses the best saved model

For more details about the model architecture and implementation, refer to the source code documentation.