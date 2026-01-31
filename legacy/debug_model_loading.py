import torch
import os

# Check what's in the checkpoint
project_root = "/home/mastermind/ielts_pred"
model_checkpoint = "src/model/bert_ielts_model_v2.pt"

checkpoint = torch.load(os.path.join(project_root, model_checkpoint), map_location='cpu')

print("="*70)
print("CHECKPOINT CONTENTS")
print("="*70)
print(f"Keys in checkpoint: {list(checkpoint.keys())}")
print(f"\nBest Val MAE: {checkpoint.get('best_val_mae', 'Not found')}")
print(f"Epoch: {checkpoint.get('epoch', 'Not found')}")
print(f"BERT model: {checkpoint.get('bert_model_name', 'Not found')}")

print("\n" + "="*70)
print("MODEL STATE DICT STRUCTURE")
print("="*70)
state_dict = checkpoint['model_state_dict']
print("\nLayer names and shapes:")
for key in sorted(state_dict.keys()):
    print(f"  {key}: {state_dict[key].shape}")

print("\n" + "="*70)
print("ARCHITECTURE DETECTION")
print("="*70)

# Check for BatchNorm vs LayerNorm
has_batch_norm = any('BatchNorm' in key or 'running_mean' in key for key in state_dict.keys())
has_layer_norm = any('LayerNorm' in key or 'layer_norm' in key for key in state_dict.keys())

print(f"Has BatchNorm layers: {has_batch_norm}")
print(f"Has LayerNorm layers: {has_layer_norm}")

# Check prediction head size
pred_head_first_layer = None
for key in state_dict.keys():
    if 'prediction_head.0.weight' in key:
        pred_head_first_layer = state_dict[key].shape
        break

print(f"Prediction head first layer shape: {pred_head_first_layer}")
if pred_head_first_layer:
    if pred_head_first_layer[0] == 128:
        print("  → This is v2 architecture (128 units)")
    elif pred_head_first_layer[0] == 256:
        print("  → This is v1 architecture (256 units)")

print("\n" + "="*70)
print("TRAINING HISTORY")
print("="*70)
if 'train_mae_history' in checkpoint:
    train_hist = checkpoint['train_mae_history']
    val_hist = checkpoint['val_mae_history']
    print(f"Training epochs: {len(train_hist)}")
    print(f"\nLast 5 epochs:")
    for i in range(max(0, len(train_hist)-5), len(train_hist)):
        print(f"  Epoch {i+1}: Train MAE={train_hist[i]:.4f} ({train_hist[i]*9:.3f} bands), "
              f"Val MAE={val_hist[i]:.4f} ({val_hist[i]*9:.3f} bands), "
              f"Gap={train_hist[i]-val_hist[i]:.4f}")
else:
    print("No training history found in checkpoint")