## IELTS Essay Scoring with BERT v3

This repository implements an IELTS essay scoring system using a BERT-based deep learning model with layer freezing (v3 architecture).

### Model Overview

The BERT v3 model is the best-performing architecture that combines:
- **DistilBERT** base model with selective layer freezing (first 3 of 6 layers)
- **10 linguistic features** extracted from essays
- **Balanced regularization** with dropout (0.35) and weight decay (0.02)
- **256→64 prediction head** with LayerNorm for stability

### Performance

- **Test MAE**: ~0.47 bands (on 0-9 scale)
- **Within ±0.5 bands**: ~78% accuracy
- **Within ±1.0 bands**: ~95% accuracy

### Repository Structure

```
IELTS_Predictions/
├── data/                      # Dataset files (not included in repo)
├── models/                    # Saved model checkpoints
│   └── bert_ielts_model_v3.pt
├── notebooks/                 # Jupyter notebooks
│   └── bert_v3_reproducible.ipynb
├── results/                   # Evaluation results
│   ├── plots/                 # Visualization outputs
│   └── metrics/               # Metric CSV files
├── src/                       # Source code
│   ├── models/                # Model definitions
│   │   ├── __init__.py
│   │   └── bert_v3.py
│   ├── data/                  # Data loading and preprocessing
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   └── features.py
│   ├── training/              # Training utilities
│   │   ├── __init__.py
│   │   └── trainer.py
│   └── evaluation/            # Evaluation and metrics
│       ├── __init__.py
│       └── metrics.py
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/pyjpg/IELTS_Predictions.git
cd IELTS_Predictions
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Dataset

The model requires an IELTS essay dataset with the following format:
- CSV file with columns: `Essay` (text) and `Overall` (band score 0-9)
- Place your dataset at: `data/ielts_writing_dataset.csv`

### Quick Start

#### Using the Jupyter Notebook (Recommended)

The easiest way to get started is with our reproducible notebook:

```bash
jupyter notebook notebooks/bert_v3_reproducible.ipynb
```

The notebook includes:
- Complete environment setup
- Data loading and preprocessing
- Model architecture explanation
- Training from scratch (or loading pre-trained)
- Comprehensive evaluation
- Inference examples

#### Training from Python

```python
import torch
from src.models import BERTIELTSScorer
from src.data import load_ielts_dataset, create_data_loaders
from src.training import train_model

# Load data
train_df, test_df = load_ielts_dataset('data/ielts_writing_dataset.csv')

# Create data loaders
train_loader, test_loader, feat_mean, feat_std = create_data_loaders(
    train_df, test_df, batch_size=4
)

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BERTIELTSScorer().to(device)

# Train
best_mae, history = train_model(
    model, train_loader, test_loader, device,
    save_path='models/bert_ielts_model_v3.pt'
)
```

#### Evaluation

```python
from src.evaluation import evaluate_model, create_evaluation_report

# Evaluate
train_metrics, train_true, train_pred = evaluate_model(model, train_loader, device)
test_metrics, test_true, test_pred = evaluate_model(model, test_loader, device)

# Create report with visualizations
create_evaluation_report(
    train_metrics, test_metrics,
    train_true, train_pred,
    test_true, test_pred
)
```

#### Inference

```python
import torch
from transformers import AutoTokenizer
from src.models import BERTIELTSScorer
from src.data import extract_linguistic_features, apply_normalization
import numpy as np

# Load model
checkpoint = torch.load('models/bert_ielts_model_v3.pt')
model = BERTIELTSScorer()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load feature normalization
feat_mean = np.load('models/bert_features_mean_v3.npy')
feat_std = np.load('models/bert_features_std_v3.npy')

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

# Score an essay
def score_essay(essay_text):
    # Tokenize
    encoding = tokenizer(
        essay_text,
        max_length=256,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Extract features
    features = extract_linguistic_features(essay_text)
    features_norm = apply_normalization(features, feat_mean, feat_std)
    features_tensor = torch.tensor(features_norm, dtype=torch.float32).unsqueeze(0)
    
    # Predict
    with torch.no_grad():
        pred_scaled = model(
            encoding['input_ids'],
            encoding['attention_mask'],
            features_tensor
        )
        band_score = (pred_scaled.item() * 9.0)
        band_score = np.clip(band_score, 1.0, 9.0)
    
    return band_score

# Example
essay = "Your essay text here..."
score = score_essay(essay)
print(f"Predicted IELTS Band Score: {score:.1f}")
```

### Model Architecture Details

**BERT v3 Hyperparameters:**
- Base model: `distilbert-base-uncased`
- Frozen layers: 3 of 6 layers
- Max sequence length: 256 tokens
- Dropout: 0.35
- Learning rate: 1.5e-5
- Weight decay: 0.02
- Batch size: 4 (with gradient accumulation of 4 steps)
- Label smoothing: 0.05
- Early stopping patience: 6 epochs

**Linguistic Features (10 total):**
1. Word count
2. Sentence count
3. Average words per sentence
4. Lexical diversity (unique word ratio)
5. Character count
6. Uppercase character ratio
7. Comma density
8. Period density
9. Average word length
10. Transition word density

### Key Design Decisions

1. **Layer Freezing**: Freezing 3 of 6 DistilBERT layers reduces memory usage (~50% fewer trainable parameters) while maintaining performance.

2. **Balanced Regularization**: The v3 model strikes a balance between capacity (v1) and regularization (v2), achieving the best generalization.

3. **Linguistic Features**: Hand-crafted features complement BERT's contextual understanding with explicit structural information.

4. **Label Smoothing**: Gentle label smoothing (0.05) helps prevent overfitting to potentially noisy labels.

### Requirements

- Python 3.8+
- PyTorch 1.12+
- Transformers 4.20+
- CUDA (optional, for GPU acceleration)

See `requirements.txt` for complete dependencies.

### Citation

If you use this code in your research, please cite:

```
@software{ielts_predictions_2024,
  title = {IELTS Essay Scoring with BERT v3},
  author = {pyjpg},
  year = {2024},
  url = {https://github.com/pyjpg/IELTS_Predictions}
}
```

### License

This project is open source and available under the MIT License.

### Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Contact

For questions or issues, please open an issue on GitHub.
