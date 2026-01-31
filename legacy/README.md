# Legacy Code

This directory contains older versions of the code that have been superseded by the clean modular architecture.

## Files

- `bert_training_v3.py` - Original monolithic training script (superseded by `src/training/trainer.py`)
- `bert_evaluation_v3.py` - Original evaluation script (superseded by `src/evaluation/metrics.py`)
- Other legacy scripts from early development

## Note

These files are kept for reference but should not be used for new development.
Use the modular code in `src/` instead, which provides better:
- Code organization
- Reusability
- Documentation
- Testing capability

For usage, see the main README.md and the Jupyter notebook in `notebooks/`.
