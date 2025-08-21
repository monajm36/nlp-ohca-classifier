# NLP OHCA Classifier

A BERT-based classifier for detecting Out-of-Hospital Cardiac Arrest (OHCA) cases in medical discharge notes using natural language processing.

## Overview

This package provides two main modules:

1. **Training Pipeline** (`ohca_training_pipeline.py`) - Complete workflow from data annotation to model training
2. **Inference Module** (`ohca_inference.py`) - Apply pre-trained models to new datasets

## Features

### Training Pipeline
- **Intelligent Sampling**: Two-stage sampling strategy (keyword-enriched + random) 
- **Annotation Interface**: Generates Excel files for manual annotation with guidelines
- **BERT-based Training**: Uses PubMedBERT optimized for medical text
- **Class Balancing**: Handles imbalanced datasets with oversampling
- **Comprehensive Evaluation**: Clinical metrics including sensitivity, specificity, PPV, NPV

### Inference Module
- **Pre-trained Model Loading**: Easy loading of trained OHCA models
- **Batch Processing**: Efficient inference on large datasets
- **Clinical Decision Support**: Probability thresholds and confidence categories
- **Quality Analysis**: Built-in tools for analyzing prediction patterns

## Installation

### Prerequisites
- Python 3.8+
- PyTorch
- CUDA (optional, for GPU acceleration)

### Install from source
```bash
git clone https://github.com/monajm36/nlp-ohca-classifier.git
cd nlp-ohca-classifier
pip install -r requirements.txt
pip install -e .
```

## Quick Start

### Training a New Model

```python
from src.ohca_training_pipeline import create_training_sample, complete_annotation_and_train
import pandas as pd

# 1. Create annotation sample
df = pd.read_csv("your_discharge_notes.csv")  # Must have: hadm_id, clean_text
annotation_df = create_training_sample(df, output_dir="./annotation_interface")

# 2. Manually annotate the Excel file (ohca_annotation.xlsx)
# Label each case: 1=OHCA, 0=Non-OHCA

# 3. Train model after annotation
results = complete_annotation_and_train(
    annotation_file="./annotation_interface/ohca_annotation.xlsx",
    model_save_path="./my_ohca_model",
    num_epochs=3
)
```

### Using a Pre-trained Model

```python
from src.ohca_inference import quick_inference
import pandas as pd

# Apply model to new data
new_data = pd.read_csv("new_discharge_notes.csv")  # Must have: hadm_id, clean_text
results = quick_inference(
    model_path="./my_ohca_model",
    data_path=new_data,
    output_path="ohca_predictions.csv"
)

# View high-confidence predictions
high_confidence = results[results['ohca_probability'] >= 0.8]
print(f"Found {len(high_confidence)} high-confidence OHCA cases")
```

## Data Format

### Input Requirements
Your CSV file must contain:
- `hadm_id`: Unique identifier for each hospital admission
- `clean_text`: Preprocessed discharge note text

### Example:
```csv
hadm_id,clean_text
12345,"Chief complaint: Cardiac arrest at home. Patient found down by family..."
12346,"Chief complaint: Chest pain. Patient presents with acute onset chest pain..."
```

### Annotation Labels
- `1`: OHCA case (cardiac arrest outside hospital)
- `0`: Non-OHCA case (everything else, including all transfer cases)

## Module Documentation

### Training Pipeline (`ohca_training_pipeline.py`)

**Main Functions:**
- `create_training_sample()` - Create balanced annotation sample
- `prepare_training_data()` - Process annotations for training
- `train_ohca_model()` - Train BERT-based classifier
- `evaluate_model()` - Comprehensive performance evaluation
- `complete_training_pipeline()` - End-to-end training workflow

**Example Usage:**
```python
from src.ohca_training_pipeline import complete_training_pipeline

# Complete training pipeline
result = complete_training_pipeline(
    data_path="discharge_notes.csv",
    annotation_dir="./annotation",
    model_save_path="./trained_model"
)
```

### Inference Module (`ohca_inference.py`)

**Main Functions:**
- `load_ohca_model()` - Load pre-trained model
- `run_inference()` - Full inference with analysis
- `quick_inference()` - Simple inference function  
- `process_large_dataset()` - Handle large datasets in chunks
- `test_model_on_sample()` - Test on specific text samples

**Example Usage:**
```python
from src.ohca_inference import run_inference, load_ohca_model

# Load model and run inference
model, tokenizer = load_ohca_model("./trained_model")
results = run_inference(model, tokenizer, new_data_df)
```

## Model Architecture

- **Base Model**: PubMedBERT (`microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract`)
- **Task**: Binary classification (OHCA vs Non-OHCA)
- **Max Sequence Length**: 512 tokens
- **Optimization**: AdamW with linear learning rate scheduling
- **Class Balancing**: Weighted loss + minority class oversampling

## Performance Metrics

The model reports comprehensive clinical metrics:
- **Sensitivity (Recall)**: Percentage of OHCA cases correctly identified
- **Specificity**: Percentage of non-OHCA cases correctly identified  
- **Precision (PPV)**: When model predicts OHCA, percentage that are correct
- **NPV**: When model predicts non-OHCA, percentage that are correct
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the receiver operating characteristic curve

## Clinical Usage

### Probability Thresholds
- **≥0.9**: Very high confidence - Priority manual review
- **0.7-0.9**: High confidence - Clinical review recommended
- **0.3-0.7**: Uncertain - Manual review suggested
- **<0.3**: Low probability - Likely non-OHCA

### Workflow Integration
1. Run inference on new discharge notes
2. Prioritize high-confidence predictions for review
3. Use medium-confidence cases for quality improvement
4. Monitor low-confidence cases for false negatives

## Repository Structure

```
nlp-ohca-classifier/
├── src/
│   ├── __init__.py
│   ├── ohca_training_pipeline.py    # Training workflow
│   └── ohca_inference.py           # Inference on new data
├── examples/
│   ├── training_example.py         # Complete training examples
│   └── inference_example.py        # Inference usage examples
├── docs/
│   └── annotation_guidelines.md    # Detailed annotation guidelines
├── requirements.txt
├── setup.py
├── README.md
└── LICENSE
```

## Examples

### Complete Training Example
```bash
cd examples
python training_example.py
```

### Inference Examples
```bash
cd examples  
python inference_example.py
```

## Advanced Usage

### Large Dataset Processing
```python
from src.ohca_inference import process_large_dataset

# Process 100K+ records in chunks
process_large_dataset(
    model_path="./trained_model",
    data_path="large_dataset.csv", 
    output_path="results.csv",
    chunk_size=5000
)
```

### Model Testing
```python
from src.ohca_inference import test_model_on_sample

# Test on specific cases
test_cases = {
    'case1': "Chief complaint: Cardiac arrest at home...",
    'case2': "Chief complaint: Chest pain, no arrest..."
}

results = test_model_on_sample("./trained_model", test_cases)
```

## Performance Benchmarks

Typical performance on validation data:
- **AUC-ROC**: 0.85-0.95
- **Sensitivity**: 85-95%
- **Specificity**: 85-95%
- **F1-Score**: 0.7-0.9

*Performance varies based on data quality and annotation consistency*

## Citation

If you use this code in your research, please cite:

```bibtex
@software{nlp_ohca_classifier,
  title={NLP OHCA Classifier: BERT-based Detection of Out-of-Hospital Cardiac Arrest in Medical Text},
  author={Mona Moukaddem},
  year={2025},
  url={https://github.com/monajm36/nlp-ohca-classifier}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Support

For questions or issues:
1. Check the [Issues](https://github.com/monajm36/nlp-ohca-classifier/issues) page
2. Create a new issue if needed
3. Review examples in the `examples/` folder

## Acknowledgments

- PubMedBERT model from Microsoft Research
- MIMIC-III dataset for model development
- Transformers library by Hugging Face
- PyTorch for deep learning framework
