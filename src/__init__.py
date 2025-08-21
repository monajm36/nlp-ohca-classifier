"""
NLP OHCA Classifier

A BERT-based classifier for detecting Out-of-Hospital Cardiac Arrest (OHCA) 
cases in medical discharge notes.
"""

from .ohca_classifier import (
    create_training_sample,
    prepare_training_data,
    train_ohca_model,
    evaluate_model,
    run_inference,
    main_pipeline,
    OHCADataset,
    OHCAInferenceDataset
)

__version__ = "1.0.0"
__author__ = "Mona Moukaddem"
__email__ = "your.email@example.com"

__all__ = [
    "create_training_sample",
    "prepare_training_data", 
    "train_ohca_model",
    "evaluate_model",
    "run_inference",
    "main_pipeline",
    "OHCADataset",
    "OHCAInferenceDataset"
]
