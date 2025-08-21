# OHCA Training Pipeline
# Complete pipeline for creating training data, annotation, and model training

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from tqdm import tqdm
import random
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight, resample
from sklearn.metrics import (
    confusion_matrix, accuracy_score, roc_auc_score, roc_curve,
    precision_recall_fscore_support
)
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

RANDOM_STATE = 42
MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seeds
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

print(f"Training Pipeline - Using device: {DEVICE}")

# =============================================================================
# STEP 1: SAMPLING FOR ANNOTATION
# =============================================================================

def create_training_sample(df, output_dir="./annotation_interface"):
    """
    Create a balanced sample for manual annotation using two-stage sampling:
    1. Keyword-enriched sampling (150 notes with 'cardiac arrest')
    2. Pure random sampling (180 notes)
    
    Args:
        df: DataFrame with columns ['hadm_id', 'clean_text']
        output_dir: Directory to save annotation interface
    
    Returns:
        DataFrame: Annotation interface with empty labels to fill
    """
    print("Creating training sample for annotation...")
    
    # Stage 1: Keyword-enriched sampling
    target_keyword = 'cardiac arrest'
    keyword_mask = df['clean_text'].str.contains(target_keyword, case=False, na=False)
    keyword_candidates = df[keyword_mask]
    
    print(f"Found {len(keyword_candidates):,} notes containing '{target_keyword}'")
    
    stage1_target = 150
    if len(keyword_candidates) >= stage1_target:
        stage1_sample = keyword_candidates.sample(n=stage1_target, random_state=RANDOM_STATE)
    else:
        remaining_needed = stage1_target - len(keyword_candidates)
        non_keyword_notes = df[~keyword_mask]
        additional_sample = non_keyword_notes.sample(n=remaining_needed, random_state=RANDOM_STATE)
        stage1_sample = pd.concat([keyword_candidates, additional_sample])
    
    stage1_sample = stage1_sample.copy()
    stage1_sample['sampling_source'] = 'keyword_enriched'
    
    # Stage 2: Random sampling
    stage2_target = 180
    already_sampled_ids = stage1_sample['hadm_id']
    remaining_notes = df[~df['hadm_id'].isin(already_sampled_ids)]
    stage2_sample = remaining_notes.sample(n=stage2_target, random_state=RANDOM_STATE+1)
    stage2_sample = stage2_sample.copy()
    stage2_sample['sampling_source'] = 'random'
    
    # Combine samples
    final_sample = pd.concat([stage1_sample, stage2_sample])
    final_sample = final_sample.drop_duplicates(subset=['hadm_id'])
    
    # Create annotation interface
    os.makedirs(output_dir, exist_ok=True)
    annotation_df = final_sample[['hadm_id', 'clean_text', 'sampling_source']].copy()
    
    # Add annotation columns
    annotation_df['ohca_label'] = ''          # 1=OHCA, 0=Non-OHCA
    annotation_df['confidence'] = ''          # 1-5 scale  
    annotation_df['notes'] = ''               # Free text reasoning
    annotation_df['annotator'] = ''           # Annotator initials
    annotation_df['annotation_date'] = ''     # Date of annotation
    
    # Randomize order
    annotation_df = annotation_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    annotation_df['annotation_order'] = range(1, len(annotation_df) + 1)
    
    # Save annotation file
    annotation_file = os.path.join(output_dir, "ohca_annotation.xlsx")
    annotation_df.to_excel(annotation_file, index=False)
    
    # Create guidelines
    guidelines_content = """
# OHCA Annotation Guidelines

## Definition
Out-of-Hospital Cardiac Arrest (OHCA) that occurred OUTSIDE a healthcare facility.

## Labels:
- **1** = OHCA (cardiac arrest outside hospital, primary reason for admission)
- **0** = Not OHCA (everything else)

## Include as OHCA (1):
- "Found down at home, CPR given"
- "Cardiac arrest at work, bystander CPR"
- "Collapsed in public, EMS resuscitation"

## Exclude as OHCA (0):
- In-hospital cardiac arrests
- History of old cardiac arrest
- Trauma/overdose causing arrest
- Chest pain without arrest

## Decision Process:
1. Did cardiac arrest happen OUTSIDE hospital? → If No: Label = 0
2. Is OHCA the PRIMARY reason for this admission? → If No: Label = 0
3. If Yes to both: Label = 1

## Confidence Scale:
- 1 = Very uncertain
- 5 = Very certain
"""
    
    guidelines_file = os.path.join(output_dir, "annotation_guidelines.md")
    with open(guidelines_file, 'w') as f:
        f.write(guidelines_content)
    
    print(f"✅ Annotation interface created:")
    print(f"  📄 File: {annotation_file}")
    print(f"  📋 Guidelines: {guidelines_file}")
    print(f"  📊 Total notes: {len(annotation_df)}")
    print(f"  🎯 Keyword-enriched: {len(stage1_sample)}")
    print(f"  🎲 Random: {len(stage2_sample)}")
    print(f"\n⚠️  Please manually annotate the Excel file before proceeding to training!")
    
    return annotation_df

# =============================================================================
# STEP 2: DATA PREPARATION FOR TRAINING
# =============================================================================

class OHCATrainingDataset(Dataset):
    """PyTorch Dataset for OHCA training"""
    
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = str(row['clean_text'])
        label = int(row['label'])
        
        # Add transfer patient prefix if applicable
        if 'transfer' in text.lower() and label == 0:
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
            'labels': torch.tensor(label, dtype=torch.long)
        }

def prepare_training_data(labeled_df):
    """
    Prepare and balance training data from manually labeled annotations
    
    Args:
        labeled_df: DataFrame with manual annotations (must have 'ohca_label' column)
    
    Returns:
        tuple: (train_dataset, val_dataset, train_df_balanced, tokenizer)
    """
    print("Preparing training data...")
    
    # Clean and prepare data
    labeled_df = labeled_df.dropna(subset=['ohca_label'])
    labeled_df['ohca_label'] = labeled_df['ohca_label'].astype(int)
    labeled_df['label'] = labeled_df['ohca_label']
    labeled_df['clean_text'] = labeled_df['clean_text'].astype(str)
    
    print(f"📊 Labeled data summary:")
    print(f"  Total cases: {len(labeled_df)}")
    print(f"  OHCA cases: {(labeled_df['label']==1).sum()}")
    print(f"  Non-OHCA cases: {(labeled_df['label']==0).sum()}")
    print(f"  OHCA prevalence: {(labeled_df['label']==1).mean():.1%}")
    
    # Split data
    if len(labeled_df) < 10:
        raise ValueError("Need at least 10 labeled cases for training")
    
    train_df, val_df = train_test_split(
        labeled_df, test_size=0.2, 
        stratify=labeled_df['label'], 
        random_state=RANDOM_STATE
    )
    
    # Balance training data (oversample minority class)
    minority = train_df[train_df['label'] == 1]
    majority = train_df[train_df['label'] == 0]
    
    if len(minority) < len(majority) and len(minority) > 0:
        target_size = min(len(majority), len(minority) * 3)  # Max 3x oversampling
        minority_upsampled = resample(
            minority, replace=True, n_samples=target_size, 
            random_state=RANDOM_STATE
        )
        train_df_balanced = pd.concat([majority, minority_upsampled])
    else:
        train_df_balanced = train_df
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Create datasets
    train_dataset = OHCATrainingDataset(train_df_balanced, tokenizer)
    val_dataset = OHCATrainingDataset(val_df, tokenizer)
    
    print(f"✅ Training data prepared:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  OHCA cases in training: {(train_df_balanced['label']==1).sum()}")
    print(f"  Non-OHCA cases in training: {(train_df_balanced['label']==0).sum()}")
    
    return train_dataset, val_dataset, train_df_balanced, tokenizer

# =============================================================================
# STEP 3: MODEL TRAINING
# =============================================================================

def train_ohca_model(train_dataset, val_dataset, train_df, tokenizer, 
                     num_epochs=3, save_path="./trained_ohca_model"):
    """
    Train OHCA classification model
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset  
        train_df: Training dataframe (for class weights)
        tokenizer: Tokenizer
        num_epochs: Number of training epochs
        save_path: Path to save trained model
    
    Returns:
        tuple: (trained_model, tokenizer)
    """
    print(f"🚀 Training OHCA model for {num_epochs} epochs...")
    
    # Initialize model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    ).to(DEVICE)
    
    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    num_training_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    
    # Class weights for balanced loss
    train_labels = train_df['label'].values
    class_weights = compute_class_weight(
        class_weight='balanced', 
        classes=np.unique(train_labels), 
        y=train_labels
    )
    weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights_tensor)
    
    print(f"⚖️  Class weights - OHCA: {class_weights[1]:.2f}, Non-OHCA: {class_weights[0]:.2f}")
    
    # Training loop
    model.train()
    all_losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in progress_bar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            epoch_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / len(train_dataloader)
        all_losses.append(avg_loss)
        print(f"📈 Epoch {epoch+1} average loss: {avg_loss:.4f}")
    
    # Save model
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    print(f"✅ Model training complete!")
    print(f"💾 Model saved to: {save_path}")
    
    return model, tokenizer

# =============================================================================
# STEP 4: MODEL EVALUATION
# =============================================================================

def evaluate_model(model, val_dataset, save_results=True, results_path="./evaluation_results.txt"):
    """
    Comprehensive model evaluation with clinical metrics
    
    Args:
        model: Trained model
        val_dataset: Validation dataset
        save_results: Whether to save results to file
        results_path: Path to save evaluation results
    
    Returns:
        dict: Comprehensive evaluation metrics
    """
    print("📊 Evaluating model performance...")
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1)
            
            predictions = torch.argmax(logits, dim=1)
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # OHCA probabilities
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Find optimal threshold
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[optimal_idx]
    
    # Calculate metrics
    optimal_preds = (all_probs >= optimal_threshold).astype(int)
    
    def calculate_metrics(y_true, y_pred):
        if len(np.unique(y_true)) < 2:
            print("⚠️  Warning: Only one class in validation set")
            return None
            
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        return {
            'accuracy': accuracy, 'precision': precision, 'recall': recall,
            'specificity': specificity, 'f1': f1, 'npv': npv,
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
        }
    
    # Calculate AUC
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.5
        print("⚠️  Warning: Could not calculate AUC")
    
    # Get metrics
    default_metrics = calculate_metrics(all_labels, all_preds)
    optimal_metrics = calculate_metrics(all_labels, optimal_preds)
    
    # Create results summary
    results_text = f"""
===============================================================================
🎯 OHCA CLASSIFIER EVALUATION RESULTS
===============================================================================

📊 Dataset Summary:
  Validation set size: {len(all_labels)}
  OHCA prevalence: {np.mean(all_labels):.1%}
  AUC-ROC: {auc:.3f}
  Optimal threshold: {optimal_threshold:.3f}

🏥 Performance with Optimal Threshold:
  Accuracy: {optimal_metrics['accuracy']:.1%}
  Sensitivity (Recall): {optimal_metrics['recall']:.1%}
  Specificity: {optimal_metrics['specificity']:.1%}
  Precision (PPV): {optimal_metrics['precision']:.1%}
  NPV: {optimal_metrics['npv']:.1%}
  F1-Score: {optimal_metrics['f1']:.3f}

📋 Confusion Matrix (Optimal Threshold):
  True Negatives (TN): {optimal_metrics['tn']}
  False Positives (FP): {optimal_metrics['fp']}
  False Negatives (FN): {optimal_metrics['fn']} 
  True Positives (TP): {optimal_metrics['tp']}

🩺 Clinical Interpretation:
  • When model predicts OHCA: {optimal_metrics['precision']:.1%} chance it's correct
  • When model predicts non-OHCA: {optimal_metrics['npv']:.1%} chance it's correct
  • Model catches {optimal_metrics['recall']:.1%} of true OHCA cases
  • Model correctly rules out {optimal_metrics['specificity']:.1%} of non-OHCA cases

⭐ Model Quality:
"""
    
    if auc >= 0.8:
        results_text += "  🟢 EXCELLENT: AUC ≥ 0.8 - Strong discriminative ability\n"
    elif auc >= 0.7:
        results_text += "  🟡 GOOD: AUC ≥ 0.7 - Acceptable discriminative ability\n"
    else:
        results_text += "  🔴 NEEDS IMPROVEMENT: AUC < 0.7 - Consider more training data\n"
    
    if optimal_metrics['f1'] >= 0.7:
        results_text += "  🟢 GOOD F1-Score: ≥ 0.7 - Well-balanced performance\n"
    elif optimal_metrics['f1'] >= 0.5:
        results_text += "  🟡 MODERATE F1-Score: ≥ 0.5 - Reasonable performance\n"
    else:
        results_text += "  🟠 LOW F1-Score: < 0.5 - Consider model improvements\n"
    
    results_text += "==============================================================================="
    
    # Print results
    print(results_text)
    
    # Save results
    if save_results:
        with open(results_path, 'w') as f:
            f.write(results_text)
        print(f"💾 Evaluation results saved to: {results_path}")
    
    return {
        'auc': auc,
        'optimal_threshold': optimal_threshold,
        'optimal_metrics': optimal_metrics,
        'default_metrics': default_metrics,
        'probabilities': all_probs,
        'predictions': optimal_preds,
        'labels': all_labels,
        'results_text': results_text
    }

# =============================================================================
# COMPLETE TRAINING PIPELINE
# =============================================================================

def complete_training_pipeline(data_path, annotation_dir="./annotation_interface", 
                              model_save_path="./trained_ohca_model"):
    """
    Complete pipeline from raw data to trained model
    
    Args:
        data_path: Path to discharge notes CSV
        annotation_dir: Directory for annotation interface
        model_save_path: Where to save the trained model
    
    Returns:
        dict: Training results and model paths
    """
    print("🚀 OHCA TRAINING PIPELINE STARTING...")
    print("="*60)
    
    # Step 1: Load data
    print("📂 Step 1: Loading discharge notes...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} discharge notes")
    
    # Step 2: Create annotation sample
    print("\n📝 Step 2: Creating annotation sample...")
    annotation_df = create_training_sample(df, annotation_dir)
    
    print("\n" + "="*60)
    print("⏸️  MANUAL ANNOTATION REQUIRED")
    print("="*60)
    print("Please complete the following steps:")
    print(f"1. Open: {annotation_dir}/ohca_annotation.xlsx")
    print(f"2. Read: {annotation_dir}/annotation_guidelines.md")
    print("3. Manually label each case (1=OHCA, 0=Non-OHCA)")
    print("4. Save the Excel file")
    print("5. Run the training continuation function")
    print("="*60)
    
    return {
        'annotation_file': f"{annotation_dir}/ohca_annotation.xlsx",
        'guidelines_file': f"{annotation_dir}/annotation_guidelines.md",
        'next_step': 'complete_annotation_and_train'
    }

def complete_annotation_and_train(annotation_file, model_save_path="./trained_ohca_model", 
                                 num_epochs=3):
    """
    Continue pipeline after manual annotation is complete
    
    Args:
        annotation_file: Path to completed annotation Excel file
        model_save_path: Where to save the trained model
        num_epochs: Number of training epochs
    
    Returns:
        dict: Complete training results
    """
    print("🔄 CONTINUING TRAINING PIPELINE...")
    print("="*60)
    
    # Step 3: Load annotations and prepare data
    print("📊 Step 3: Loading annotations and preparing data...")
    labeled_df = pd.read_excel(annotation_file)
    train_dataset, val_dataset, train_df, tokenizer = prepare_training_data(labeled_df)
    
    # Step 4: Train model
    print("\n🏋️ Step 4: Training model...")
    model, tokenizer = train_ohca_model(
        train_dataset, val_dataset, train_df, tokenizer, 
        num_epochs=num_epochs, save_path=model_save_path
    )
    
    # Step 5: Evaluate model
    print("\n📈 Step 5: Evaluating model...")
    results = evaluate_model(
        model, val_dataset, 
        save_results=True, 
        results_path=f"{model_save_path}/evaluation_results.txt"
    )
    
    print("\n✅ TRAINING PIPELINE COMPLETE!")
    print(f"📁 Model saved to: {model_save_path}")
    print(f"📊 Results saved to: {model_save_path}/evaluation_results.txt")
    
    return {
        'model_path': model_save_path,
        'evaluation_results': results,
        'model': model,
        'tokenizer': tokenizer
    }

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("OHCA Training Pipeline")
    print("="*30)
    print("This module provides complete training pipeline for OHCA classification.")
    print("\nMain functions:")
    print("• create_training_sample() - Create annotation interface")
    print("• prepare_training_data() - Prepare training datasets") 
    print("• train_ohca_model() - Train the model")
    print("• evaluate_model() - Evaluate performance")
    print("• complete_training_pipeline() - Full pipeline")
    print("\nSee examples/ folder for detailed usage examples.")
