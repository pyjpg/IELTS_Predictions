# Models Directory

This directory contains the trained BERT v3 model checkpoint.

## File Structure

- `bert_ielts_model_v3.pt` - The trained BERT v3 model (600MB)
  - Contains model state dict, optimizer state, training history
  - Best validation MAE: ~0.052 (scaled 0-1) = ~0.47 bands

## Feature Normalization Files

These files are needed for inference and are generated during training:
- `bert_features_mean_v3.npy` - Mean values for feature normalization
- `bert_features_std_v3.npy` - Standard deviation for feature normalization

## Note

The model checkpoint file (`.pt`) is excluded from git due to its large size.
To use the model:
1. Train it using the notebook or training script
2. Or download a pre-trained checkpoint if available

## Model Details

- **Architecture**: DistilBERT base with 3 frozen layers
- **Parameters**: ~34M trainable (out of ~67M total)
- **Input**: Essay text + 10 linguistic features
- **Output**: IELTS band score (0-9 scale)
