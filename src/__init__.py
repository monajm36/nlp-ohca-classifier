"""
NLP OHCA Classifier

A BERT-based classifier for detecting Out-of-Hospital Cardiac Arrest (OHCA) 
cases in medical discharge notes.

This package contains two main modules:

1. ohca_training_pipeline: Complete training pipeline from annotation to model training
2. ohca_inference: Apply pre-trained models to new datasets
"""

# Training pipeline imports
from .ohca_training_pipeline import (
    create_training_sample,
    prepare_training_data,
    train_ohca_model,
    evaluate_model,
    complete_training_pipeline,
    complete_annotation_and_train,
    OHCATrainingDataset
)

# Inference imports  
from .ohca_inference import (
    load_ohca_model,
    run_inference,
    quick_inference,
    process_large_dataset,
    test_model_on_sample,
    get_high_confidence_cases,
    analyze_predictions,
    OHCAInferenceDataset
)

__version__ = "1.0.0"
__author__ = "Mona Moukaddem"
__email__ = "your.email@example.com"

# Training pipeline functions
__training_functions__ = [
    "create_training_sample",
    "prepare_training_data", 
    "train_ohca_model",
    "evaluate_model",
    "complete_training_pipeline",
    "complete_annotation_and_train",
    "OHCATrainingDataset"
]

# Inference functions
__inference_functions__ = [
    "load_ohca_model",
    "run_inference",
    "quick_inference", 
    "process_large_dataset",
    "test_model_on_sample",
    "get_high_confidence_cases",
    "analyze_predictions",
    "OHCAInferenceDataset"
]

__all__ = __training_functions__ + __inference_functions__
