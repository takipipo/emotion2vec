# Downstream task for emotion2vec
We provide training scripts for the IEMOCAP dataset with standard 5531 utterances for 4 emotions (happy, sad, angry, neutral) to evaluate the performance of emotion2vec.

## Data processing and feature extraction

### Prepare from scratch
1. Download IEMOCAP dataset from official website [IEMOCAP_full_release](https://docs.google.com/forms/d/e/1FAIpQLScBecgI2K5bFTrXi_-05IYSSwOcqL5mX7dh57xcJV1m_NoznA/viewform?usp=sf_link).

2. Parpare features and labels.
```bash
# set your own path here
IEMOCAP_ROOT=/path/to/IEMOCAP_full_release
manifest_path=/path/to/manifest
checkpoint_path=/path/to/emotion2vec_ckpt
feat_path=/path/to/feats
fairseq_root=/path/to/fairseq_root
model_path=/path/to/emotion2vec_model_file

# generate manifest and labels from raw dataset
bash scripts/iemocap_manifest_and_labels.sh $IEMOCAP_ROOT $manifest_path

# generate features with emotion2vec
bash scripts/emotion2vec_extract_features.sh $fairseq_root $manifest_path $model_path $checkpoint_path $feat_path

cp $manifest_path/train.emo $feat_path
```
You will get 3 files in `$feat_path`:
- `train.npy`
- `train.lengths`
- `train.emo`

### Download from Google Drive
You can also download the processed data from:
- `train.npy` [Google Drive]() | [Baidu Netdisk](https://pan.baidu.com/s/1kWpT2X5gVc6pYULN0WdSdg?pwd=hsjt) (password: hsjt)
- `train.lengths` [Google Drive]() | [Baidu Netdisk](https://pan.baidu.com/s/1pa7GHnGyTZw_fi-U1y4YgA?pwd=a99c) (password: a99c)
- `train.emo` [Google Drive]() | [Baidu Netdisk](https://pan.baidu.com/s/1A_DTIoC7VbTxly5HzUZfpw?pwd=j9xv) (password: j9xv)

## Train a downstream model
We provide scripts for training a downstream model with only linear layers. We take leave-one-session-out 5-fold cross-validation as an example.
```bash
feat_path=/path/to/feats
save_dir=/path/to/save_dir
bash train.sh ${feat_path}/train $save_dir
```
As frame-level features are provided, you can implement more complex downstream models for further  performance improvements.