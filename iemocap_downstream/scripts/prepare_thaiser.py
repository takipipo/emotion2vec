import pandas as pd
import os
import torch
import tqdm
from npy_append_array import NpyAppendArray
from emotion2vec_speech_features import Emotion2vecFeatureReader
def prepare_emotion2vec_features(df, model_file, checkpoint, layer, output_dir, data_root, split="train"):
    """
    Prepare emotion2vec features from a DataFrame containing PATH and EMOTION columns
    
    Args:
        df: pandas DataFrame with PATH and EMOTION columns
        model_file: path to emotion2vec model file
        checkpoint: path to checkpoint
        layer: which layer to use
        output_dir: directory to save output files
        data_root: root directory for audio files
        split: split name (train/val/test)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize feature reader
    reader = Emotion2vecFeatureReader(model_file, checkpoint, layer)
    
    # Create output files
    save_path = os.path.join(output_dir, split)
    
    # Remove existing .npy file if it exists
    if os.path.exists(save_path + ".npy"):
        os.remove(save_path + ".npy")
    
    # Create NpyAppendArray for features
    npaa = NpyAppendArray(save_path + ".npy")
    
    # Create .emo file for emotions
    with open(save_path + ".emo", "w") as emo_f:
        # Create .lengths file for feature lengths
        with open(save_path + ".lengths", "w") as len_f:
            # Process each audio file
            for idx, row in tqdm.tqdm(df.iterrows(), total=len(df)):
                # Get the filename without extension and path
                filename = os.path.splitext(os.path.basename(row['PATH']))[0]
                
                # Convert relative path to absolute path
                abs_path = os.path.abspath(os.path.join(data_root, row['PATH'].lstrip('./dataset/')))
                
                try:
                    # Extract features
                    d2v_feats = reader.get_feats(abs_path)
                    
                    # Write feature length
                    print(len(d2v_feats), file=len_f)
                    
                    # Write emotion label
                    print(f"{filename}\t{row['EMOTION'].lower()}", file=emo_f)
                    
                    # Append features to .npy file
                    if len(d2v_feats) > 0:
                        npaa.append(d2v_feats.numpy())
                        
                except Exception as e:
                    print(f"Error processing {abs_path}: {str(e)}")
                    continue

def main():
    # Read your pre-split DataFrames
    train_df = pd.read_csv('../../dataset/df_train.csv')
    val_df = pd.read_csv('../../dataset/df_val.csv')
    test_df = pd.read_csv('../../dataset/df_test.csv')
    
    # Define parameters
    model_file = "../../upstream"
    checkpoint = "../../upstream/emotion2vec_base.pt"
    layer = 12
    output_dir = "../../dataset"
    data_root = "../../dataset"  # Adjust this to point to your root directory
    
    # Prepare features for each split
    prepare_emotion2vec_features(train_df, model_file, checkpoint, layer, output_dir, data_root, "train")
    prepare_emotion2vec_features(val_df, model_file, checkpoint, layer, output_dir, data_root, "val")
    prepare_emotion2vec_features(test_df, model_file, checkpoint, layer, output_dir, data_root, "test")

if __name__ == "__main__":
    main()