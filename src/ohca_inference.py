# OHCA Inference Module
# Apply pre-trained OHCA classifier to new datasets

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Inference Module - Using device: {DEVICE}")

# =============================================================================
# INFERENCE DATASET CLASS
# =============================================================================

class OHCAInferenceDataset(Dataset):
    """Dataset for OHCA inference on new data"""
    
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Validate required columns
        if 'hadm_id' not in self.data.columns or 'clean_text' not in self.data.columns:
            raise ValueError("DataFrame must contain 'hadm_id' and 'clean_text' columns")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = str(row['clean_text'])
        
        # Apply preprocessing consistent with training
        if 'transfer' in text.lower():
            text = "TRANSFERRED_PATIENT " + text
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'hadm_id': row['hadm_id']
        }

# =============================================================================
# MODEL LOADING FUNCTIONS
# =============================================================================

def load_ohca_model(model_path):
    """
    Load pre-trained OHCA model and tokenizer
    
    Args:
        model_path: Path to saved model directory
    
    Returns:
        tuple: (model, tokenizer)
    """
    print(f"📂 Loading OHCA model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    
    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model = model.to(DEVICE)
        model.eval()
        
        print("✅ Model loaded successfully")
        print(f"   Device: {DEVICE}")
        print(f"   Model type: {type(model).__name__}")
        
        return model, tokenizer
        
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")

# =============================================================================
# INFERENCE FUNCTIONS
# =============================================================================

def run_inference(model, tokenizer, inference_df, batch_size=16, 
                 output_path=None, probability_threshold=0.5):
    """
    Run OHCA inference on new data
    
    Args:
        model: Pre-trained OHCA model
        tokenizer: Model tokenizer
        inference_df: DataFrame with columns ['hadm_id', 'clean_text']
        batch_size: Batch size for inference
        output_path: Optional path to save results CSV
        probability_threshold: Threshold for binary predictions
    
    Returns:
        DataFrame: Results with probabilities and predictions
    """
    print(f"🔍 Running OHCA inference on {len(inference_df):,} cases...")
    
    # Validate input data
    required_cols = ['hadm_id', 'clean_text']
    missing_cols = [col for col in required_cols if col not in inference_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Remove any rows with missing data
    clean_df = inference_df.dropna(subset=required_cols).copy()
    if len(clean_df) < len(inference_df):
        print(f"⚠️  Removed {len(inference_df) - len(clean_df)} rows with missing data")
    
    # Create dataset and dataloader
    inference_dataset = OHCAInferenceDataset(clean_df, tokenizer)
    inference_dataloader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False)
    
    # Run inference
    model.eval()
    all_probabilities = []
    all_hadm_ids = []
    
    with torch.no_grad():
        for batch in tqdm(inference_dataloader, desc="Processing batches"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            hadm_ids = batch['hadm_id']
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = F.softmax(outputs.logits, dim=1)
            
            # Get OHCA probabilities (class 1)
            ohca_probs = probs[:, 1].cpu().numpy()
            
            all_probabilities.extend(ohca_probs)
            all_hadm_ids.extend(hadm_ids)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'hadm_id': all_hadm_ids,
        'ohca_probability': all_probabilities
    })
    
    # Add predictions with different thresholds
    results_df['prediction_050'] = (results_df['ohca_probability'] >= 0.5).astype(int)
    results_df['prediction_070'] = (results_df['ohca_probability'] >= 0.7).astype(int)
    results_df['prediction_090'] = (results_df['ohca_probability'] >= 0.9).astype(int)
    results_df['prediction_custom'] = (results_df['ohca_probability'] >= probability_threshold).astype(int)
    
    # Add confidence categories
    def categorize_confidence(prob):
        if prob >= 0.9:
            return "Very High"
        elif prob >= 0.7:
            return "High"
        elif prob >= 0.3:
            return "Medium"
        elif prob >= 0.1:
            return "Low"
        else:
            return "Very Low"
    
    results_df['confidence_category'] = results_df['ohca_probability'].apply(categorize_confidence)
    
    # Sort by probability (highest first)
    results_df = results_df.sort_values('ohca_probability', ascending=False).reset_index(drop=True)
    
    # Print summary
    print(f"\n📊 Inference Results Summary:")
    print(f"   Total cases processed: {len(results_df):,}")
    print(f"   Mean OHCA probability: {results_df['ohca_probability'].mean():.4f}")
    print(f"   Max OHCA probability: {results_df['ohca_probability'].max():.3f}")
    print(f"   Min OHCA probability: {results_df['ohca_probability'].min():.3f}")
    
    # Probability distribution
    print(f"\n🎯 Probability Distribution:")
    thresholds = [0.9, 0.8, 0.7, 0.6, 0.5, 0.3, 0.1]
    for threshold in thresholds:
        count = (results_df['ohca_probability'] >= threshold).sum()
        pct = count / len(results_df) * 100
        print(f"   ≥{threshold}: {count:,} cases ({pct:.2f}%)")
    
    # Confidence categories
    print(f"\n📈 Confidence Distribution:")
    conf_dist = results_df['confidence_category'].value_counts()
    for category, count in conf_dist.items():
        pct = count / len(results_df) * 100
        print(f"   {category}: {count:,} cases ({pct:.1f}%)")
    
    # Save results if path provided
    if output_path:
        results_df.to_csv(output_path, index=False)
        print(f"\n💾 Results saved to: {output_path}")
    
    return results_df

def get_high_confidence_cases(results_df, threshold=0.8, max_cases=100):
    """
    Extract high-confidence OHCA predictions for manual review
    
    Args:
        results_df: Results from run_inference()
        threshold: Minimum probability threshold
        max_cases: Maximum number of cases to return
    
    Returns:
        DataFrame: High-confidence cases sorted by probability
    """
    high_conf = results_df[results_df['ohca_probability'] >= threshold].copy()
    high_conf = high_conf.head(max_cases)
    
    print(f"🎯 Found {len(high_conf)} high-confidence cases (≥{threshold})")
    
    return high_conf

def analyze_predictions(results_df, original_df=None):
    """
    Analyze prediction patterns and provide clinical insights
    
    Args:
        results_df: Results from run_inference()
        original_df: Optional original dataframe to merge with results
    
    Returns:
        dict: Analysis summary
    """
    print("📋 Analyzing prediction patterns...")
    
    # Basic statistics
    stats = {
        'total_cases': len(results_df),
        'mean_probability': results_df['ohca_probability'].mean(),
        'std_probability': results_df['ohca_probability'].std(),
        'median_probability': results_df['ohca_probability'].median(),
        'high_confidence_cases': (results_df['ohca_probability'] >= 0.8).sum(),
        'predicted_ohca_050': results_df['prediction_050'].sum(),
        'predicted_ohca_070': results_df['prediction_070'].sum(),
        'predicted_ohca_090': results_df['prediction_090'].sum(),
    }
    
    # Confidence distribution
    conf_dist = results_df['confidence_category'].value_counts().to_dict()
    
    # Print analysis
    print(f"\n📊 Prediction Analysis:")
    print(f"   Total cases: {stats['total_cases']:,}")
    print(f"   Mean probability: {stats['mean_probability']:.4f}")
    print(f"   Predicted OHCA (≥0.5): {stats['predicted_ohca_050']:,}")
    print(f"   High confidence (≥0.8): {stats['high_confidence_cases']:,}")
    
    if stats['predicted_ohca_050'] > 0:
        prevalence = stats['predicted_ohca_050'] / stats['total_cases'] * 100
        print(f"   Estimated OHCA prevalence: {prevalence:.2f}%")
    
    # Clinical recommendations
    print(f"\n🏥 Clinical Recommendations:")
    if stats['high_confidence_cases'] > 0:
        print(f"   • Priority review: {stats['high_confidence_cases']} high-confidence cases")
    if stats['predicted_ohca_070'] > 0:
        print(f"   • Clinical review: {stats['predicted_ohca_070']} cases ≥0.7 probability")
    
    uncertain_cases = ((results_df['ohca_probability'] >= 0.3) & 
                      (results_df['ohca_probability'] < 0.7)).sum()
    if uncertain_cases > 0:
        print(f"   • Manual review suggested: {uncertain_cases} uncertain cases")
    
    return {
        'statistics': stats,
        'confidence_distribution': conf_dist,
        'high_confidence_cases': results_df[results_df['ohca_probability'] >= 0.8]
    }

# =============================================================================
# BATCH PROCESSING FUNCTIONS
# =============================================================================

def process_large_dataset(model_path, data_path, output_path, 
                         chunk_size=10000, batch_size=16):
    """
    Process large datasets in chunks to avoid memory issues
    
    Args:
        model_path: Path to trained model
        data_path: Path to input CSV file
        output_path: Path for output results
        chunk_size: Number of rows per chunk
        batch_size: Batch size for inference
    
    Returns:
        str: Path to completed results file
    """
    print(f"🔄 Processing large dataset in chunks of {chunk_size:,}...")
    
    # Load model once
    model, tokenizer = load_ohca_model(model_path)
    
    # Read data in chunks
    chunk_results = []
    chunk_num = 0
    
    for chunk_df in pd.read_csv(data_path, chunksize=chunk_size):
        chunk_num += 1
        print(f"\n📦 Processing chunk {chunk_num} ({len(chunk_df):,} rows)...")
        
        # Run inference on chunk
        chunk_result = run_inference(
            model, tokenizer, chunk_df, 
            batch_size=batch_size, output_path=None
        )
        
        chunk_results.append(chunk_result)
        
        # Save intermediate results
        temp_path = f"{output_path}.chunk_{chunk_num}.csv"
        chunk_result.to_csv(temp_path, index=False)
        print(f"💾 Chunk {chunk_num} saved to: {temp_path}")
    
    # Combine all chunks
    print(f"\n🔗 Combining {len(chunk_results)} chunks...")
    final_results = pd.concat(chunk_results, ignore_index=True)
    
    # Sort by probability and save
    final_results = final_results.sort_values('ohca_probability', ascending=False)
    final_results.to_csv(output_path, index=False)
    
    print(f"✅ Complete results saved to: {output_path}")
    print(f"📊 Total cases processed: {len(final_results):,}")
    
    # Clean up intermediate files
    for i in range(1, chunk_num + 1):
        temp_path = f"{output_path}.chunk_{i}.csv"
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    return output_path

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_inference(model_path, data_path, output_path=None):
    """
    Quick inference function for simple use cases
    
    Args:
        model_path: Path to trained model
        data_path: Path to input CSV (or DataFrame)
        output_path: Optional output path
    
    Returns:
        DataFrame: Inference results
    """
    print("🚀 Quick OHCA Inference")
    
    # Load model
    model, tokenizer = load_ohca_model(model_path)
    
    # Load data
    if isinstance(data_path, str):
        df = pd.read_csv(data_path)
        print(f"📂 Loaded {len(df):,} cases from {data_path}")
    else:
        df = data_path.copy()
        print(f"📊 Processing {len(df):,} cases from DataFrame")
    
    # Run inference
    results = run_inference(model, tokenizer, df, output_path=output_path)
    
    # Quick summary
    ohca_cases = (results['ohca_probability'] >= 0.5).sum()
    high_conf = (results['ohca_probability'] >= 0.8).sum()
    
    print(f"\n✅ Quick Summary:")
    print(f"   Predicted OHCA cases: {ohca_cases:,}")
    print(f"   High confidence: {high_conf:,}")
    
    return results

def test_model_on_sample(model_path, sample_texts):
    """
    Test model on a few sample texts for quick validation
    
    Args:
        model_path: Path to trained model
        sample_texts: List of text strings or dict with hadm_id: text
    
    Returns:
        DataFrame: Test results
    """
    print("🧪 Testing model on sample texts...")
    
    # Prepare test data
    if isinstance(sample_texts, dict):
        test_df = pd.DataFrame([
            {'hadm_id': hadm_id, 'clean_text': text} 
            for hadm_id, text in sample_texts.items()
        ])
    else:
        test_df = pd.DataFrame([
            {'hadm_id': f'TEST_{i:03d}', 'clean_text': text} 
            for i, text in enumerate(sample_texts, 1)
        ])
    
    # Run inference
    model, tokenizer = load_ohca_model(model_path)
    results = run_inference(model, tokenizer, test_df, output_path=None)
    
    # Print results
    print(f"\n🔍 Test Results:")
    for _, row in results.iterrows():
        prob = row['ohca_probability']
        pred = "OHCA" if prob >= 0.5 else "Non-OHCA"
        conf = row['confidence_category']
        
        print(f"   {row['hadm_id']}: {pred} (prob={prob:.3f}, {conf})")
        
        # Show text preview
        text_preview = test_df[test_df['hadm_id']==row['hadm_id']]['clean_text'].iloc[0]
        print(f"      Text: {text_preview[:100]}...")
        print()
    
    return results

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("OHCA Inference Module")
    print("="*25)
    print("This module provides inference capabilities for pre-trained OHCA models.")
    print("\nMain functions:")
    print("• load_ohca_model() - Load pre-trained model")
    print("• run_inference() - Run inference on new data")
    print("• quick_inference() - Simple inference function")
    print("• process_large_dataset() - Handle large datasets")
    print("• test_model_on_sample() - Test on sample texts")
    print("\nSee examples/ folder for detailed usage examples.")
